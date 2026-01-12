import numpy as np
import logging
import random
import math
from enum import Enum, auto
from collections import deque
from dataclasses import dataclass

# [SCI Integration] Import New MBD Physics & Router
from physics import TrainConvoy, DavisResistanceModel
from router import IntelligentRouter

# 配置日志
logger = logging.getLogger("Edge.Vehicle")


# =========================================================================
# [Layer 0] Advanced Control Algorithms (MPC)
# =========================================================================

class LongitudinalMPC:
    """
    [Control Logic] Linear Time-Varying Model Predictive Control (LTV-MPC).

    Solves the optimal control problem:
        min J = sum(w_v * (v - v_ref)^2 + w_u * u^2 + w_du * (u - u_prev)^2)
    s.t.
        v_{k+1} = A_k * v_k + B_k * u_k + D_k (Linearized Dynamics)
        -1 <= u <= 1

    Why MPC?
    Unlike PID, MPC anticipates future resistance (e.g. mud) and smooths out
    control actions to save energy (minimize u^2) and reduce coupler stress (minimize du^2).
    """

    def __init__(self, dt=0.1, horizon=10):
        self.dt = dt
        self.H = horizon  # Prediction Horizon (steps)
        self.last_u = 0.0

        # Tuning Weights (Pareto Weights)
        self.w_v = 10.0  # Penalty for velocity error
        self.w_u = 0.1  # Penalty for energy consumption
        self.w_du = 50.0  # Penalty for jerk (smoothness)

    def solve(self, v_curr, v_ref, mass_kg, mud_factor):
        """
        Analytical solution for unconstrained MPC (clipped post-hoc).
        """
        # 1. Linearize Vehicle Dynamics around v_curr
        # F_net = F_traction * u - F_resistance(v)
        # v_dot = (F_max * u - (A + B*v + C*v^2)) / m
        # Linearize R(v) approx R(v0) + R'(v0)*(v - v0)

        # Physics constants (Simplified from physics.py for internal model)
        A_drag = 2000.0 + 10000.0 * mud_factor  # Rolling + Mud base
        B_drag = 50.0
        C_drag = 0.5 * 1.225 * 10.0 * 0.5  # Aerodynamic
        F_max = 100000.0  # Approx max tractive force (Newtons)

        # R(v) = A + Bv + Cv^2
        # dR/dv = B + 2Cv
        R0 = A_drag + B_drag * abs(v_curr) + C_drag * (v_curr ** 2)
        dR_dv = B_drag + 2 * C_drag * abs(v_curr)

        # State Space: v_{k+1} = a * v_k + b * u_k + d
        # v_dot = (F_max/m)*u - (R0 + dR*(v - v0))/m
        # v_dot = (-dR/m)*v + (F_max/m)*u + (dR*v0 - R0)/m

        # Discrete matrices (Euler)
        m = max(mass_kg, 1000.0)
        a = 1.0 - (dR_dv / m) * self.dt
        b = (F_max / m) * self.dt
        d = ((dR_dv * v_curr - R0) / m) * self.dt

        # 2. Receding Horizon Optimization (Simplified Gradient Descent for 1D)
        # Since this is a scalar system, we can use a heuristic or a fast analytical solver.
        # For simplicity and robustness in this simulation loop, we implement a
        # "Predictive Functional Control" approach:
        # Assume u is constant over the horizon (or decays), find best u_0.

        # We iterate over candidate control inputs to find the minimum cost
        # This is a robust numerical way to solve the 1D constrained QP.
        candidates = np.linspace(-1.0, 1.0, 21)  # Discretize throttle
        best_u = self.last_u
        min_cost = float('inf')

        for u_test in candidates:
            cost = 0.0
            v_pred = v_curr

            # Smoothness penalty immediately
            cost += self.w_du * (u_test - self.last_u) ** 2

            # Predict future
            for k in range(self.H):
                v_pred = a * v_pred + b * u_test + d
                cost += self.w_v * (v_pred - v_ref) ** 2
                cost += self.w_u * (u_test ** 2)

            if cost < min_cost:
                min_cost = cost
                best_u = u_test

        # 3. Update State
        self.last_u = best_u
        return best_u


# =========================================================================
# [Layer 1] State & Mode Definitions
# =========================================================================

class ControlMode(Enum):
    """
    [Control Mode]
    STIGMERGY_FOLLOW: Follow the holographic flow field (Hauler default).
    ACTIVE_SCOUTING:  Explore and update risk maps (Scout default).
    PHYSICS_FALLBACK: Degradation mode when Cyber layer fails (Self-Healing).
    EMERGENCY_STOP:   Mechanical failure.
    """
    STIGMERGY_FOLLOW = auto()
    ACTIVE_SCOUTING = auto()
    PHYSICS_FALLBACK = auto()  # Cyber-Resilience
    EMERGENCY_STOP = auto()


class VehicleState(Enum):
    IDLE = auto()
    MISSION_OP = auto()  # Loading/Unloading
    MOVING = auto()
    WAITING_INTERLOCK = auto()  # Waiting for switch confirmation
    RECOVERY = auto()  # Self-healing process


@dataclass
class EnergyAudit:
    traction_joules: float = 0.0
    compute_joules: float = 0.0
    comm_joules: float = 0.0

    @property
    def total(self):
        return self.traction_joules + self.compute_joules + self.comm_joules


# =========================================================================
# [Layer 2] Cyber-Physical Agent Implementation (Multi-Body Edition)
# =========================================================================

class VehicleAgent:
    def __init__(self, agent_id, train_type_name, global_config, start_node, map_graph, infra_agents):
        self.id = agent_id
        self.cfg = global_config
        self.env = global_config['environment']
        self.map = map_graph
        self.infra = infra_agents

        # [Perception] Injected by main.py later
        self.all_vehicles = []

        # [Heterogeneity] Role Definition
        train_cfg = global_config['train_configurations'][train_type_name]
        self.is_scout = (train_cfg.get('role') == 'scout')

        # --- 1. Physics Kernel (MBD) ---
        self.physics = TrainConvoy(train_type_name, global_config)

        # --- 2. Advanced Control (MPC) ---
        # Initialize MPC solver with 2.0s horizon (20 steps * 0.1s)
        # Note: dt passed to MPC should match simulation dt or be a multiple
        sim_dt = global_config['simulation']['dt']
        self.controller = LongitudinalMPC(dt=sim_dt, horizon=20)

        # [Fix: Odometer Anchor] Records physics position at edge entry
        self.edge_entry_odometer = 0.0

        # --- 3. Navigation ---
        if start_node in self.map.nodes:
            self.pos_2d = np.array(self.map.nodes[start_node].pos, dtype=float)
            # [Fix: Anti-Clipping Spawn] Add micro-jitter to prevent 0-distance locking
            self.pos_2d += np.random.uniform(-1.0, 1.0, size=2)
        else:
            self.pos_2d = np.array([0.0, 0.0])

        self.current_node_id = start_node
        self.previous_node_id = None  # [Fix: Memory for Anti-U-Turn]
        self.next_node_id = None
        self.target_node = self._assign_target()
        self.router = IntelligentRouter(map_graph)
        self.path_queue = deque()

        # --- 4. State ---
        self.state = VehicleState.IDLE
        self.mode = ControlMode.ACTIVE_SCOUTING if self.is_scout else ControlMode.STIGMERGY_FOLLOW
        self.current_speed = 0.0
        self.comm_health = 1.0
        self.trust_status = 1.0
        self.energy = EnergyAudit()
        self.last_telemetry = {}

        # [Fix: Watchdog & Startup Timers]
        self.stuck_timer = 0.0
        self.startup_timer = 0.0

    def step(self, dt, global_time):
        """[Main Loop] Sense -> Plan -> MPC -> Act"""
        # 1. Perception
        self._update_comm_health(global_time)
        if self.comm_health < 0.2 and self.mode != ControlMode.PHYSICS_FALLBACK:
            logger.warning(f"[{self.id}] Comm Loss! Downgrading.")
            self.mode = ControlMode.PHYSICS_FALLBACK

        # [Fix: Startup Timer for Depot Exit]
        # Allow vehicles time to clear the depot without triggering AEB immediately
        self.startup_timer += dt
        is_startup_phase = self.startup_timer < 15.0  # Grace period

        # 2. Planning
        # [Fix: Ensure target exists via Watchdog]
        if not self.next_node_id:
            if self.state == VehicleState.IDLE:
                self._handle_idle_logic(dt)
            elif self.state == VehicleState.MOVING:
                if self.mode == ControlMode.PHYSICS_FALLBACK:
                    self._plan_fallback_crawl()
                elif self.is_scout:
                    self._plan_active_scouting(global_time)
                else:
                    self._plan_flow_following(global_time)

            # [Fix: Logic-Compliant Deadlock Recovery]
            # If we STILL have no next_node_id after planning, we are stuck.
            if not self.next_node_id:
                self.stuck_timer += dt
                if self.stuck_timer > 5.0:  # 5s timeout
                    # DO NOT change target. Force re-planning to same target.
                    # Clearing queue forces Router to find a NEW path from scratch.
                    logger.warning(f"[{self.id}] Stuck. Forcing re-planning to {self.target_node}.")
                    self.path_queue.clear()
                    self.stuck_timer = 0.0
            else:
                self.stuck_timer = 0.0

        # 3. Control (MPC + AEB)
        throttle_cmd = 0.0
        dynamics = None

        # Only move physically if we have a valid route
        if self.next_node_id:
            self.state = VehicleState.MOVING
            v_ref = self._get_mode_speed_limit()

            # [Fix: Enhanced AEB with Angle Check & Startup Exemption]
            safe_dist = 40.0
            collision_risk = False

            # Only enable AEB after startup phase to allow clearing depot
            if not is_startup_phase:
                my_target_pos = np.array(self.map.nodes[self.next_node_id].pos)
                my_pos = self.pos_2d
                # Heading vector
                heading = my_target_pos - my_pos
                heading_norm = np.linalg.norm(heading)
                if heading_norm > 0.1:
                    heading /= heading_norm
                else:
                    heading = np.array([1.0, 0.0])

                for other in self.all_vehicles:
                    if other.id == self.id: continue

                    vec_to_other = other.pos_2d - my_pos
                    dist = np.linalg.norm(vec_to_other)

                    if dist < safe_dist:
                        # Angle check: Only brake if obstacle is IN FRONT (>45deg cone)
                        to_other_dir = vec_to_other / (dist + 0.001)
                        angle_cos = np.dot(heading, to_other_dir)

                        if angle_cos > 0.7:
                            v_ref = 0.0  # EMERGENCY BRAKE
                            collision_risk = True
                            break

                            # [Fix: Startup Boost]
            # If in startup phase, force speed to separate overlapping vehicles
            if is_startup_phase:
                v_ref = max(v_ref, 8.0)

            self._trigger_switch(global_time)

            # [MPC Execution]
            total_mass = getattr(self.physics, 'mass_total', 5000.0)
            local_mud = self.env.get('mud_factor', 0.5)

            throttle_cmd = self.controller.solve(
                v_curr=self.current_speed,
                v_ref=v_ref,
                mass_kg=total_mass,
                mud_factor=local_mud
            )

            # 4. Actuation (Physics)
            dynamics = self.physics.step_rk4(dt, throttle_cmd, local_mud)
            self.current_speed = dynamics['loco_vel']

            # [Fix: Sync Logic]
            self._sync_position(dynamics)

            self.energy.traction_joules += abs(dynamics['motor_current'] * 400.0) * dt

        else:
            # Idle Physics (Brake)
            throttle_cmd = -1.0 if abs(self.current_speed) > 0.1 else 0.0
            dynamics = self.physics.step_rk4(dt, throttle_cmd, self.env.get('mud_factor', 0.5))
            self.current_speed = dynamics['loco_vel']
            # Don't update pos_2d if idle

        # 5. Feedback
        self._broadcast_state(global_time)

        # Return proper telemetry even if idle
        if not dynamics:
            dynamics = {
                'loco_vel': 0.0, 'loco_pos': 0.0, 'motor_current': 0.0,
                'coupler_force_1': 0.0, 'mu_effective': 0.0
            }

        return self._pack_telemetry(dynamics)

    def _sync_position(self, dynamics):
        """
        [Coordinate Drift Fix + Overshoot Protection]
        """
        if not self.next_node_id:
            return  # Stay at current node

        # 1. Get Geometry
        p_start = np.array(self.map.nodes[self.current_node_id].pos)
        p_end = np.array(self.map.nodes[self.next_node_id].pos)
        edge_vec = p_end - p_start
        edge_len = np.linalg.norm(edge_vec)

        if edge_len < 0.1:
            self.current_node_id = self.next_node_id
            self.next_node_id = None
            return

        # 2. Calculate Progress based on Physics
        current_abs_pos = dynamics['loco_pos']
        dist_on_edge = current_abs_pos - self.edge_entry_odometer

        progress = dist_on_edge / edge_len

        # 3. Update 2D Position
        if progress >= 1.0:
            # Arrival Logic
            self.pos_2d = p_end

            # [Fix: Update History to prevent U-Turn]
            self.previous_node_id = self.current_node_id

            self.current_node_id = self.next_node_id
            self.next_node_id = None

            # [CRITICAL FIX] Update Anchor: Consume edge length
            self.edge_entry_odometer += edge_len
            self.stuck_timer = 0.0  # Reset watchdog

            # Instant Re-plan for smoothness
            if self.is_scout:
                self._plan_active_scouting(0)
            else:
                self._plan_flow_following(0)
        else:
            self.pos_2d = p_start + edge_vec * progress

    def _plan_active_scouting(self, now):
        self.router.step(now)
        if not self.path_queue and not self.next_node_id:
            path = self.router.get_dynamic_path(self.current_node_id, self.target_node, self)
            self.path_queue = deque(path)

            # Remove current node from path if present
            if self.path_queue and self.path_queue[0] == self.current_node_id:
                self.path_queue.popleft()

            if self.path_queue:
                self.next_node_id = self.path_queue.popleft()

            # [Fix: Fallback] If pathfinding failed (e.g. cold start), re-roll target
            if not self.next_node_id:
                self.target_node = self._assign_target()

        local_mud = self.env.get('mud_factor', 0.5)
        if local_mud > 0.6:
            infra = self.infra.get(self.current_node_id)
            if infra:
                infra.flow_field.inject_event(mass=5.0, velocity=0.0)

    def _plan_flow_following(self, now):
        """
        [Algorithm] Stigmergy-based Navigation.
        Follows the gradient of the Holographic Flow Field.
        """
        if self.next_node_id: return

        all_neighbors = list(self.map.graph.neighbors(self.current_node_id))
        if not all_neighbors: return  # Dead end

        # [Fix: Anti-Backtracking Logic]
        # Only consider nodes that are NOT where we just came from.
        # This forces the train to keep moving forward in the graph.
        candidates = [n for n in all_neighbors if n != self.previous_node_id]

        # If dead end (only neighbor is previous), we MUST reverse.
        if not candidates:
            candidates = all_neighbors

        best_next = None
        min_cost = float('inf')

        current_pos = np.array(self.map.nodes[self.current_node_id].pos)
        target_pos = np.array(self.map.nodes[self.target_node].pos)
        # Euclidean distance remaining
        current_dist = np.linalg.norm(target_pos - current_pos)

        for n in candidates:
            infra = self.infra.get(n)
            p = infra.flow_field.get_state()['potential'] if infra else 0.0

            pos_n = np.array(self.map.nodes[n].pos)
            dist_n = np.linalg.norm(pos_n - target_pos)

            # Cost = Potential * Weight + Distance
            # If Potential is 0 (Cold Start), this becomes Distance-only (A* heuristic)
            cost = p * 10.0 + dist_n

            # Soft Penalty for moving away from target (Anti-U-Turn heuristic 2)
            if dist_n > current_dist:
                cost += 50.0

            if cost < min_cost:
                min_cost = cost
                best_next = n

        if best_next:
            self.next_node_id = best_next
        else:
            self.next_node_id = random.choice(candidates)

    def _plan_fallback_crawl(self):
        if self.next_node_id: return
        neighbors = list(self.map.graph.neighbors(self.current_node_id))
        if neighbors: self.next_node_id = random.choice(neighbors)

    def _trigger_switch(self, now):
        if self.mode == ControlMode.PHYSICS_FALLBACK: return
        infra = self.infra.get(self.next_node_id)
        dist = np.linalg.norm(self.pos_2d - np.array(self.map.nodes[self.next_node_id].pos))
        if dist < 60.0 and infra:
            curr_p = self.map.nodes[self.current_node_id].pos
            next_p = self.map.nodes[self.next_node_id].pos
            vec = np.array(next_p) - np.array(curr_p)
            req = 'REVERSE' if abs(vec[1]) > abs(vec[0]) else 'NORMAL'
            infra.handle_hardware_signal({'type': 'SWITCH_REQ', 'target_state': req})

    def _get_mode_speed_limit(self):
        if self.mode == ControlMode.PHYSICS_FALLBACK:
            return 2.0
        elif self.mode == ControlMode.ACTIVE_SCOUTING:
            return 12.0
        return 10.0

    def _update_comm_health(self, now):
        infra = self.infra.get(self.current_node_id)
        self.comm_health = 1.0 if (infra and infra.state.name != 'STALLED') else 0.5

    def _broadcast_state(self, now):
        if self.mode == ControlMode.PHYSICS_FALLBACK: return
        infra = self.infra.get(self.current_node_id)
        if infra:
            packet = self._build_semantic_packet(now)
            resp = infra.handle_semantic_packet(packet)
            if resp.get('status') == 'REJECTED_UNTRUSTED':
                self.trust_status *= 0.9

    def _build_semantic_packet(self, now):
        return {
            'vid': self.id,
            'pos': self.pos_2d,
            'vel': self.current_speed,
            'timestamp': now,
            'pos_uncertainty': 0.5,
            'eta': now + 5.0,
            'duration': 3.0,
            'mud_detected': self.env.get('mud_factor', 0.5)
        }

    def _handle_idle_logic(self, dt):
        if not self.target_node:
            self.target_node = self._assign_target()

        if self.target_node and self.current_node_id != self.target_node:
            self.state = VehicleState.MOVING

    def _assign_target(self):
        nodes = list(self.map.nodes.keys())
        if not nodes: return None
        # Don't pick current node
        opts = [n for n in nodes if n != self.current_node_id]
        return random.choice(opts) if opts else nodes[0]

    def _pack_telemetry(self, dynamics):
        # [Fix: Export Unit Offsets for Viz]
        unit_offsets = []
        # Reconstruct relative positions from physics.units
        current_offset = 0.0
        for i, unit in enumerate(self.physics.units):
            if i > 0:
                gap = 1.0  # Standard coupler
                prev = self.physics.units[i - 1]
                current_offset += (prev.length / 2 + gap + unit.length / 2)
            unit_offsets.append(current_offset)

        self.last_telemetry = {
            'id': self.id,
            'state': self.state.name,
            'mode': self.mode.name,
            'vel': self.current_speed,
            'energy': self.energy.total,
            'comm_health': self.comm_health,
            'current': dynamics.get('motor_current', 0.0),
            'force': dynamics.get('coupler_force_1', 0.0),
            'mass': getattr(self.physics, 'mass_total', 5000),
            'mu': dynamics.get('mu_effective', 0.0),
            'unit_offsets': unit_offsets
        }
        return self.last_telemetry


class MaliciousVehicle(VehicleAgent):
    """
    [Adversary] Implements False Data Injection Attacks (FDIA).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_scout = True
        self.mode = ControlMode.ACTIVE_SCOUTING
        logger.warning(f"⚠️ [Security] Malicious Agent {self.id} Activated!")

    def _plan_active_scouting(self, now):
        super()._plan_active_scouting(now)
        if self.env.get('mud_factor', 0.5) > 0.6:
            infra = self.infra.get(self.current_node_id)
            if infra:
                infra.flow_field.inject_event(mass=-2.0, velocity=10.0)

    def _build_semantic_packet(self, now):
        packet = super()._build_semantic_packet(now)
        packet['mud_detected'] = 0.1
        if self.env.get('mud_factor', 0.5) > 0.6:
            packet['vel'] = 30.0
        return packet