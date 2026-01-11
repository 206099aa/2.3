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
    STIGMERGY_FOLLOW = auto()
    ACTIVE_SCOUTING = auto()
    PHYSICS_FALLBACK = auto()
    EMERGENCY_STOP = auto()


class VehicleState(Enum):
    IDLE = auto()
    MISSION_OP = auto()
    MOVING = auto()
    WAITING_INTERLOCK = auto()
    RECOVERY = auto()


@dataclass
class EnergyAudit:
    traction_joules: float = 0.0
    compute_joules: float = 0.0
    comm_joules: float = 0.0

    @property
    def total(self):
        return self.traction_joules + self.compute_joules + self.comm_joules


# =========================================================================
# [Layer 2] Cyber-Physical Agent Implementation
# =========================================================================

class VehicleAgent:
    def __init__(self, agent_id, train_type_name, global_config, start_node, map_graph, infra_agents):
        self.id = agent_id
        self.cfg = global_config
        self.env = global_config['environment']
        self.map = map_graph
        self.infra = infra_agents

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

        # --- 3. Navigation ---
        if start_node in self.map.nodes:
            self.pos_2d = np.array(self.map.nodes[start_node].pos, dtype=float)
        else:
            self.pos_2d = np.array([0.0, 0.0])

        self.current_node_id = start_node
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

    def step(self, dt, global_time):
        """[Main Loop] Sense -> Plan -> MPC -> Act"""
        # 1. Perception
        self._update_comm_health(global_time)
        if self.comm_health < 0.2 and self.mode != ControlMode.PHYSICS_FALLBACK:
            logger.warning(f"[{self.id}] Comm Loss! Downgrading.")
            self.mode = ControlMode.PHYSICS_FALLBACK

        # 2. Planning
        if self.state == VehicleState.IDLE:
            self._handle_idle_logic(dt)
        elif self.state == VehicleState.MOVING:
            if self.mode == ControlMode.PHYSICS_FALLBACK:
                self._plan_fallback_crawl()
            elif self.is_scout:
                self._plan_active_scouting(global_time)
            else:
                self._plan_flow_following(global_time)

        # 3. Control (MPC)
        throttle_cmd = 0.0
        if self.state == VehicleState.MOVING:
            v_ref = self._get_mode_speed_limit()
            if self.next_node_id: self._trigger_switch(global_time)

            # [MPC Execution]
            # Estimate total mass for the internal model
            total_mass = sum([u.mass for u in self.physics.units])
            local_mud = self.env.get('mud_factor', 0.5)

            throttle_cmd = self.controller.solve(
                v_curr=self.current_speed,
                v_ref=v_ref,
                mass_kg=total_mass,
                mud_factor=local_mud
            )

        # 4. Actuation (Physics)
        local_mud = self.env.get('mud_factor', 0.5)
        dynamics = self.physics.step_rk4(dt, throttle_cmd, local_mud)

        self.current_speed = dynamics['loco_vel']
        self._sync_position(dt)

        # 5. Feedback
        self._broadcast_state(global_time)
        self.energy.traction_joules += abs(dynamics['motor_current'] * 400.0) * dt

        return self._pack_telemetry(dynamics)

    # ... [Rest of the Planning Methods remain similar but concise] ...

    def _plan_active_scouting(self, now):
        self.router.step(now)
        if not self.path_queue and not self.next_node_id:
            path = self.router.get_dynamic_path(self.current_node_id, self.target_node, self)
            self.path_queue = deque(path)
            if self.path_queue and self.path_queue[0] == self.current_node_id:
                self.path_queue.popleft()
            self.next_node_id = self.path_queue[0] if self.path_queue else None

        local_mud = self.env.get('mud_factor', 0.5)
        if local_mud > 0.6:
            infra = self.infra.get(self.current_node_id)
            if infra:
                infra.flow_field.inject_event(mass=5.0, velocity=0.0)

    def _plan_flow_following(self, now):
        if self.next_node_id: return
        neighbors = list(self.map.graph.neighbors(self.current_node_id))
        best_next = None
        min_potential = float('inf')
        for n in neighbors:
            infra = self.infra.get(n)
            p = infra.flow_field.get_state()['potential'] if infra else 0.0
            dist = np.linalg.norm(np.array(self.map.nodes[n].pos) - np.array(self.map.nodes[self.target_node].pos))
            cost = p * 2.0 + dist
            if cost < min_potential:
                min_potential = cost
                best_next = n
        self.next_node_id = best_next

    def _plan_fallback_crawl(self):
        if self.next_node_id: return
        neighbors = list(self.map.graph.neighbors(self.current_node_id))
        if not neighbors: return
        best = min(neighbors, key=lambda n: np.linalg.norm(
            np.array(self.map.nodes[n].pos) - np.array(self.map.nodes[self.target_node].pos)
        ))
        self.next_node_id = best

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
        if random.random() < 0.01:
            self.state = VehicleState.MOVING
            self.target_node = self._assign_target()

    def _assign_target(self):
        return random.choice(list(self.map.nodes.keys()))

    def _sync_position(self, dt):
        if self.next_node_id:
            target_pos = np.array(self.map.nodes[self.next_node_id].pos)
            vec = target_pos - self.pos_2d
            dist = np.linalg.norm(vec)
            if dist > 0.5:
                self.pos_2d += (vec / dist) * self.current_speed * dt
            else:
                self.current_node_id = self.next_node_id
                self.next_node_id = None

    def _pack_telemetry(self, dynamics):
        self.last_telemetry = {
            'id': self.id,
            'state': self.state.name,
            'mode': self.mode.name,
            'vel': self.current_speed,
            'energy': self.energy.total,
            'comm_health': self.comm_health,
            'current': dynamics['motor_current'],
            'force': dynamics['coupler_force_1'],
            'mass': dynamics.get('total_mass', 5000),
            'mu': dynamics['mu_effective']
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