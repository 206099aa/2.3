import numpy as np
import logging
import random
import math
import networkx as nx
from enum import Enum, auto
from collections import deque
from dataclasses import dataclass

# [SCI Integration] Import New MBD Physics & Router
from physics import TrainConvoy
from router import IntelligentRouter

# 配置日志
logger = logging.getLogger("Edge.Vehicle")


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
        self.cfg = global_config  # Keep full config for flexibility
        self.env = global_config['environment']
        self.map = map_graph
        self.infra = infra_agents

        # [Heterogeneity] Determine Role from Train Configuration
        # Now based on the 'role' field in train_configurations
        train_cfg = global_config['train_configurations'][train_type_name]
        self.is_scout = (train_cfg.get('role') == 'scout')

        # --- 1. Physics Kernel (Advanced Multi-Body Dynamics) ---
        # Initialize the convoy (Loco + Wagons + Couplers)
        self.physics = TrainConvoy(train_type_name, global_config)

        # --- 2. Positioning & Navigation ---
        if start_node in self.map.nodes:
            self.pos_2d = np.array(self.map.nodes[start_node].pos, dtype=float)
        else:
            self.pos_2d = np.array([0.0, 0.0])

        self.current_node_id = start_node
        self.next_node_id = None
        self.target_node = self._assign_target()

        # --- 3. Cyber-Physical Router ---
        self.router = IntelligentRouter(map_graph)
        self.path_queue = deque()

        # --- 4. Control State ---
        self.state = VehicleState.IDLE
        self.mode = ControlMode.ACTIVE_SCOUTING if self.is_scout else ControlMode.STIGMERGY_FOLLOW

        self.current_speed = 0.0
        self.comm_health = 1.0  # 1.0=Good, 0.0=Disconnected
        self.trust_status = 1.0

        # --- 5. Telemetry ---
        self.energy = EnergyAudit()
        self.last_telemetry = {}

    def step(self, dt, global_time):
        """
        [Main Loop]
        1. Sense (Physics & Cyber)
        2. Plan (Stigmergy or Scouting)
        3. Act (MPC -> Throttle -> MBD)
        """
        # --- Phase 1: Perception & Self-Diagnosis ---
        self._update_comm_health(global_time)

        # [Self-Healing] Check for Cyber-Failure
        if self.comm_health < 0.2:
            if self.mode != ControlMode.PHYSICS_FALLBACK:
                logger.warning(f"[{self.id}] Comm Loss! Downgrading to PHYSICS_FALLBACK.")
                self.mode = ControlMode.PHYSICS_FALLBACK

        # --- Phase 2: Strategic Planning (Heterogeneous) ---
        if self.state == VehicleState.IDLE:
            self._handle_idle_logic(dt)

        elif self.state == VehicleState.MOVING:
            if self.mode == ControlMode.PHYSICS_FALLBACK:
                # [Resilience] Crawl home using only local sensors
                self._plan_fallback_crawl()
            elif self.is_scout:
                # [Heterogeneity] Scouts actively update the map
                self._plan_active_scouting(global_time)
            else:
                # [Stigmergy] Haulers follow the flow field (Gradient Descent)
                self._plan_flow_following(global_time)

        # --- Phase 3: Control Execution ---
        throttle_cmd = 0.0  # -1.0 (Brake) to 1.0 (Power)

        if self.state == VehicleState.MOVING:
            # Determine reference speed based on mode
            v_ref = self._get_mode_speed_limit()

            # Switch Interlocking (Edge Interaction)
            if self.next_node_id:
                self._trigger_switch(global_time)

            # MPC Control (Now outputs Throttle/Brake)
            throttle_cmd = self._mpc_control(dt, v_ref)

        # --- Phase 4: Physics Integration (MBD) ---
        # Pass environmental mud factor to the physics engine
        local_mud = self.env.get('mud_factor', 0.5)

        # Step the Multi-Body System
        dynamics = self.physics.step_rk4(dt, throttle_cmd, local_mud)

        self.current_speed = dynamics['loco_vel']
        self._sync_position(dt)

        # --- Phase 5: Cyber-Physical Feedback ---
        self._broadcast_state(global_time)

        # Energy Audit (Power = Voltage * Current approx)
        # Using simplified power estimation from physics feedback
        p_inst = abs(dynamics['motor_current'] * 400.0)  # Assume 400V bus
        self.energy.traction_joules += p_inst * dt

        return self._pack_telemetry(dynamics)

    # =========================================================================
    # [Module 1] Heterogeneous Planning Strategies
    # =========================================================================

    def _plan_active_scouting(self, now):
        """[Role: Scout] Active Sensing & Risk Injection."""
        # 1. Update router
        self.router.step(now)

        # 2. Check path
        if not self.path_queue and not self.next_node_id:
            path = self.router.get_dynamic_path(self.current_node_id, self.target_node, self)
            self.path_queue = deque(path)
            if self.path_queue and self.path_queue[0] == self.current_node_id:
                self.path_queue.popleft()
            self.next_node_id = self.path_queue[0] if self.path_queue else None

        # 3. [Active Sensing]
        local_mud = self.env.get('mud_factor', 0.5)
        if local_mud > 0.6:
            infra = self.infra.get(self.current_node_id)
            if infra:
                # Scout injects a "Virtual Mass" into the flow field
                packet = self._build_semantic_packet(now)
                # This call is implicit in the simulation, representing V2I
                infra.flow_field.inject_event(mass=5.0, velocity=0.0)

    def _plan_flow_following(self, now):
        """[Role: Hauler] Gradient Descent on Holographic Field."""
        if self.next_node_id: return

        neighbors = list(self.map.graph.neighbors(self.current_node_id))
        best_next = None
        min_potential = float('inf')

        for n in neighbors:
            # Query Edge Node State (Stigmergy)
            infra = self.infra.get(n)
            p = 0.0
            if infra:
                state = infra.flow_field.get_state()
                p = state['potential']

            dist = np.linalg.norm(np.array(self.map.nodes[n].pos) - np.array(self.map.nodes[self.target_node].pos))
            # Cost Function: Heavily weight potential (Risk)
            cost = p * 2.0 + dist

            if cost < min_potential:
                min_potential = cost
                best_next = n

        self.next_node_id = best_next

    def _plan_fallback_crawl(self):
        """[Self-Healing] Physics-Only Mode."""
        if self.next_node_id: return
        neighbors = list(self.map.graph.neighbors(self.current_node_id))
        if not neighbors: return
        # Geometric heuristic only
        best = min(neighbors, key=lambda n: np.linalg.norm(
            np.array(self.map.nodes[n].pos) - np.array(self.map.nodes[self.target_node].pos)
        ))
        self.next_node_id = best

    # =========================================================================
    # [Module 2] Control & Actuation
    # =========================================================================

    def _trigger_switch(self, now):
        """Send Switch Request."""
        if self.mode == ControlMode.PHYSICS_FALLBACK: return

        infra = self.infra.get(self.next_node_id)
        dist = np.linalg.norm(self.pos_2d - np.array(self.map.nodes[self.next_node_id].pos))

        if dist < 60.0 and infra:  # Increased trigger distance for longer trains
            curr_p = self.map.nodes[self.current_node_id].pos
            next_p = self.map.nodes[self.next_node_id].pos
            vec = np.array(next_p) - np.array(curr_p)
            req = 'REVERSE' if abs(vec[1]) > abs(vec[0]) else 'NORMAL'

            infra.handle_hardware_signal({
                'type': 'SWITCH_REQ',
                'target_state': req
            })

    def _mpc_control(self, dt, v_ref):
        """
        [Control] Adaptive Cruise Control.
        Outputs: Throttle (-1.0 to 1.0)
        """
        # 1. Physics-Based Speed Profiling (Mud Awareness)
        local_mud = self.env.get('mud_factor', 0.5)
        # Slower target in mud to maintain adhesion margin
        v_safe_mud = 15.0 * (1.0 - 0.6 * local_mud)
        v_target = min(v_ref, v_safe_mud)

        # 2. Simple P-Controller -> Throttle
        err = v_target - self.current_speed

        # Gain Scheduling: Haulers need higher gain due to huge mass
        kp = 0.5 if self.is_scout else 0.8

        throttle = kp * err
        return np.clip(throttle, -1.0, 1.0)

    def _get_mode_speed_limit(self):
        if self.mode == ControlMode.PHYSICS_FALLBACK:
            return 2.0
        elif self.mode == ControlMode.ACTIVE_SCOUTING:
            return 12.0
        return 10.0  # Hauler cruising speed

    # =========================================================================
    # [Module 3] Cyber-Physical Support
    # =========================================================================

    def _update_comm_health(self, now):
        infra = self.infra.get(self.current_node_id)
        if infra:
            if infra.state.name == 'STALLED':
                self.comm_health = 0.0
            else:
                self.comm_health = 1.0
        else:
            self.comm_health = 0.5

    def _broadcast_state(self, now):
        if self.mode == ControlMode.PHYSICS_FALLBACK: return
        infra = self.infra.get(self.current_node_id)
        if infra:
            packet = self._build_semantic_packet(now)
            resp = infra.handle_semantic_packet(packet)
            if resp.get('status') == 'REJECTED_UNTRUSTED':
                self.trust_status *= 0.9
                logger.warning(f"[{self.id}] Trusted dropped by infra!")

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
        nodes = list(self.map.nodes.keys())
        return random.choice(nodes)

    def _sync_position(self, dt):
        if self.next_node_id:
            target_pos = np.array(self.map.nodes[self.next_node_id].pos)
            vec = target_pos - self.pos_2d
            dist = np.linalg.norm(vec)
            if dist > 0.5:  # Tolerance
                direction = vec / dist
                self.pos_2d += direction * self.current_speed * dt
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
            'force': dynamics['coupler_force_1'],  # Lead coupler force
            'mass': dynamics.get('total_mass', 5000),  # Total convoy mass
            'mu': dynamics['mu_effective']
        }
        return self.last_telemetry


# =========================================================================
# [Adversary] Malicious Vehicle (FDIA)
# =========================================================================

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
        """Lure Attack: Report SAFE when DANGEROUS."""
        super()._plan_active_scouting(now)
        real_mud = self.env.get('mud_factor', 0.5)

        if real_mud > 0.6:
            infra = self.infra.get(self.current_node_id)
            if infra:
                # Inject 'negative mass' or low risk to lure others
                # Simulating a "Safety Signal"
                infra.flow_field.inject_event(mass=-2.0, velocity=10.0)
                logger.info(f"[Attack] {self.id} injecting FALSE SAFE signal.")

    def _build_semantic_packet(self, now):
        """Anomaly Attack: Report Impossible Physics."""
        packet = super()._build_semantic_packet(now)

        # 1. Lie about environment
        packet['mud_detected'] = 0.1

        # 2. Lie about velocity (Physics Violation)
        if self.env.get('mud_factor', 0.5) > 0.6:
            packet['vel'] = 30.0  # Impossible speed in mud

        return packet