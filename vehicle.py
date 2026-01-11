import numpy as np
import logging
import random
import math
import networkx as nx
from enum import Enum, auto
from collections import deque
from dataclasses import dataclass

# [SCI Integration] Import Physics & Router
from physics import RailVehicleMBDSystem
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
# [Layer 2] Cyber-Physical Agent Implementation
# =========================================================================

class VehicleAgent:
    def __init__(self, agent_id, vehicle_type_cfg, env_config, start_node, map_graph, infra_agents):
        self.id = agent_id
        self.cfg = vehicle_type_cfg
        self.env = env_config
        self.map = map_graph
        self.infra = infra_agents

        # [Heterogeneity] Determine Role based on config
        # Scouts are light and fast; Haulers are heavy
        self.is_scout = "Scout" in agent_id or self.cfg.get('mass_full', 5000) < 4000

        # --- 1. Physics Kernel ---
        self.physics = RailVehicleMBDSystem(self.cfg, self.env)

        # --- 2. Positioning & Navigation ---
        if start_node in self.map.nodes:
            self.pos_2d = np.array(self.map.nodes[start_node].pos, dtype=float)
            self.physics._init_position(spacing=2.0)
        else:
            self.pos_2d = np.array([0.0, 0.0])

        self.current_node_id = start_node
        self.next_node_id = None
        self.target_node = self._assign_target()

        # --- 3. Cyber-Physical Router ---
        # Each vehicle carries an onboard router instance (Edge Computing)
        self.router = IntelligentRouter(map_graph)
        self.path_queue = deque()

        # --- 4. Control State ---
        self.state = VehicleState.IDLE
        self.mode = ControlMode.ACTIVE_SCOUTING if self.is_scout else ControlMode.STIGMERGY_FOLLOW

        self.current_speed = 0.0
        self.mpc_prev_u = 0.0
        self.comm_health = 1.0  # 1.0=Good, 0.0=Disconnected
        self.trust_status = 1.0  # My reputation in the network

        # --- 5. Telemetry ---
        self.energy = EnergyAudit()
        self.last_telemetry = {}

    def step(self, dt, global_time):
        """
        [Main Loop]
        1. Sense (Physics & Cyber)
        2. Plan (Stigmergy or Scouting)
        3. Act (MPC & Switch Control)
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
        u_cmd = 0.0
        if self.state == VehicleState.MOVING:
            # Determine reference speed based on mode
            v_ref = self._get_mode_speed_limit()

            # Switch Interlocking (Edge Interaction)
            if self.next_node_id:
                self._trigger_switch(global_time)

            # MPC Control
            u_cmd = self._mpc_control(dt, v_ref)

        # --- Phase 4: Physics Integration ---
        dynamics = self.physics.step_rk4(dt, u_cmd)
        self.current_speed = dynamics['loco_vel']
        self._sync_position(dt)

        # --- Phase 5: Cyber-Physical Feedback ---
        # Broadcast Semantic Packet (Position + Intent)
        self._broadcast_state(global_time)

        # Energy Audit
        p_inst = abs(u_cmd * dynamics['motor_current'])
        self.energy.traction_joules += p_inst * dt

        return self._pack_telemetry(dynamics)

    # =========================================================================
    # [Module 1] Heterogeneous Planning Strategies
    # =========================================================================

    def _plan_active_scouting(self, now):
        """
        [Role: Scout]
        1. Detect high-mud areas (Physics Sensor).
        2. Send RISK_UPDATE to Edge Nodes (Cyber Injection).
        """
        # 1. Update own router
        self.router.step(now)

        # 2. Check path
        if not self.path_queue and not self.next_node_id:
            # Scouts use A* to find global optimal to explore
            path = self.router.get_dynamic_path(self.current_node_id, self.target_node, self)
            self.path_queue = deque(path)
            if self.path_queue and self.path_queue[0] == self.current_node_id:
                self.path_queue.popleft()
            self.next_node_id = self.path_queue[0] if self.path_queue else None

        # 3. [Active Sensing] If current edge is muddy, warn the infrastructure
        # Simulation: Read local mud from map
        local_mud = self.env.get('mud_factor', 0.5)
        # (In real robot, this comes from wheel slip sensor)

        if local_mud > 0.6:
            # Send High-Priority Risk Update
            infra = self.infra.get(self.current_node_id)
            if infra:
                # Scout injects a "Virtual Mass" into the flow field to deter others
                # This modifies the 'potential' in infrastructure.py
                packet = {
                    'vid': self.id,
                    'type': 'RISK_UPDATE',
                    'mud_detected': local_mud,
                    'timestamp': now
                }
                # Implicitly handled by flow injection in infra
                infra.flow_field.inject_event(mass=5.0, velocity=0.0)  # High pressure!

    def _plan_flow_following(self, now):
        """
        [Role: Hauler]
        Greedy Gradient Descent on the Holographic Flow Field.
        Avoids high-potential (congested/muddy) nodes.
        """
        if self.next_node_id: return

        # Look at all neighbors
        neighbors = list(self.map.graph.neighbors(self.current_node_id))
        best_next = None
        min_potential = float('inf')

        for n in neighbors:
            # Query Edge Node State (Simulating V2I)
            infra = self.infra.get(n)
            p = 0.0
            if infra:
                state = infra.flow_field.get_state()
                p = state['potential']

            # Add distance heuristic (A* like)
            dist = np.linalg.norm(np.array(self.map.nodes[n].pos) - np.array(self.map.nodes[self.target_node].pos))

            # Cost = Flow Potential + Distance Heuristic
            # Haulers play it safe: Weight Potential heavily
            cost = p * 2.0 + dist

            if cost < min_potential:
                min_potential = cost
                best_next = n

        self.next_node_id = best_next

    def _plan_fallback_crawl(self):
        """
        [Self-Healing] Physics-Only Mode.
        Ignore all cyber signals. Move forward slowly if no physical obstacle.
        """
        if self.next_node_id: return

        # Simple heuristic: Pick any neighbor that gets closer to target
        neighbors = list(self.map.graph.neighbors(self.current_node_id))
        if not neighbors: return

        # Just pick the geometrically closest one
        best = min(neighbors, key=lambda n: np.linalg.norm(
            np.array(self.map.nodes[n].pos) - np.array(self.map.nodes[self.target_node].pos)
        ))
        self.next_node_id = best

    # =========================================================================
    # [Module 2] Control & Actuation
    # =========================================================================

    def _trigger_switch(self, now):
        """Send Switch Request (Cyber)"""
        if self.mode == ControlMode.PHYSICS_FALLBACK:
            return  # Cannot switch in fallback mode! Must wait or stop.

        infra = self.infra.get(self.next_node_id)
        dist = np.linalg.norm(self.pos_2d - np.array(self.map.nodes[self.next_node_id].pos))

        if dist < 40.0 and infra:
            # Calculate required state
            curr_p = self.map.nodes[self.current_node_id].pos
            next_p = self.map.nodes[self.next_node_id].pos
            vec = np.array(next_p) - np.array(curr_p)
            req = 'REVERSE' if abs(vec[1]) > abs(vec[0]) else 'NORMAL'

            infra.handle_hardware_signal({
                'type': 'SWITCH_REQ',
                'target_state': req
            })

    def _mpc_control(self, dt, v_ref):
        # 1. Safety Check (Physics Sensor)
        # Radar/Lidar check for physical obstacles
        safe_dist = 100.0
        # ... (Simplified obstacle check logic)

        # 2. Physics-Based Speed Profiling
        # Slow down if mud is high (Self-Preservation)
        local_mud = self.env.get('mud_factor', 0.5)
        v_safe_mud = 15.0 * (1.0 - 0.5 * local_mud)
        v_target = min(v_ref, v_safe_mud)

        # 3. Simple P-Controller (MPC simplified for code length)
        # u = Kp * error + FF
        err = v_target - self.current_speed
        kp = 2000.0 if self.is_scout else 5000.0  # Heavy haulers need more gain
        u = kp * err

        return np.clip(u, -48.0, 48.0)  # Voltage limit

    def _get_mode_speed_limit(self):
        if self.mode == ControlMode.PHYSICS_FALLBACK:
            return 2.0  # Crawl speed (Safe)
        elif self.mode == ControlMode.ACTIVE_SCOUTING:
            return 12.0  # Fast exploration
        return 8.0  # Efficient hauling

    # =========================================================================
    # [Module 3] Cyber-Physical Support
    # =========================================================================

    def _update_comm_health(self, now):
        # Simulate RSSI check
        # In full sim, this would query comms.py channel model
        infra = self.infra.get(self.current_node_id)
        if infra:
            # If node is STALLED, comms are likely down too
            if infra.state.name == 'STALLED':
                self.comm_health = 0.0
            else:
                self.comm_health = 1.0
        else:
            self.comm_health = 0.5  # Weak signal in open field

    def _broadcast_state(self, now):
        """
        Normal semantic broadcast.
        (MaliciousVehicle will override this to inject False Data)
        """
        if self.mode == ControlMode.PHYSICS_FALLBACK: return

        infra = self.infra.get(self.current_node_id)
        if infra:
            # 标准真实数据包
            packet = self._build_semantic_packet(now)

            # Infrastructure applies TrustFilter here
            resp = infra.handle_semantic_packet(packet)

            # React to rejection
            if resp.get('status') == 'REJECTED_UNTRUSTED':
                self.trust_status *= 0.9  # Degradation
                logger.warning(f"[{self.id}] Trusted dropped by infra!")

    def _build_semantic_packet(self, now):
        """Helper to construct packet, making it easier to override."""
        return {
            'vid': self.id,
            'pos': self.pos_2d,
            'vel': self.current_speed,
            'timestamp': now,
            'pos_uncertainty': self.network_uncertainty,
            'eta': now + 5.0,  # Estimation
            'duration': 3.0,
            'mud_detected': self.env.get('mud_factor', 0.5)  # [Normal] Report True mud
        }

    def _handle_idle_logic(self, dt):
        # Simple mission generator
        if random.random() < 0.01:
            self.state = VehicleState.MOVING
            self.target_node = self._assign_target()

    def _assign_target(self):
        # Random valid node
        nodes = list(self.map.nodes.keys())
        return random.choice(nodes)

    def _sync_position(self, dt):
        # Integrate velocity to update 2D position
        if self.next_node_id:
            target_pos = np.array(self.map.nodes[self.next_node_id].pos)
            vec = target_pos - self.pos_2d
            dist = np.linalg.norm(vec)
            if dist > 0.1:
                direction = vec / dist
                self.pos_2d += direction * self.current_speed * dt
            else:
                self.current_node_id = self.next_node_id
                self.next_node_id = None  # Arrived at node

    def _pack_telemetry(self, dynamics):
        self.last_telemetry = {
            'id': self.id,
            'state': self.state.name,
            'mode': self.mode.name,
            'vel': self.current_speed,
            'energy': self.energy.total,
            'comm_health': self.comm_health,
            'current': dynamics['motor_current'],
            'force': dynamics['coupler_force_1']
        }
        return self.last_telemetry


# =========================================================================
# [New Class] Malicious Adversary (FDIA)
# =========================================================================

class MaliciousVehicle(VehicleAgent):
    """
    [Adversary] Implements False Data Injection Attacks (FDIA).
    Inherits from VehicleAgent but overrides communication to inject deceptive data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 伪装成 Scout 以便四处游荡并散布虚假信息
        self.is_scout = True
        self.mode = ControlMode.ACTIVE_SCOUTING
        logger.warning(f"⚠️ [Security] Malicious Agent {self.id} Activated!")

    def _plan_active_scouting(self, now):
        """
        [Attack Type 1: Lure / Safety Injection]
        Override scouting logic to report 'Safe' (mud=0.1) even in dangerous areas.
        """
        # 1. 执行正常的寻路逻辑 (为了看起来像个正常车)
        super()._plan_active_scouting(now)

        # 2. 执行攻击：如果当前环境恶劣，反而发送“路况良好”的信号
        # 这会降低全息流场中该节点的势能，诱骗其他车辆进入拥堵/泥泞区
        real_mud = self.env.get('mud_factor', 0.5)

        # 只有在真实现象糟糕时才攻击，制造反差
        if real_mud > 0.6:
            infra = self.infra.get(self.current_node_id)
            if infra:
                # 构造虚假包
                fake_packet = {
                    'vid': self.id,
                    'type': 'RISK_UPDATE',
                    'mud_detected': 0.1,  # <--- LIE: Report clean road
                    'timestamp': now
                }
                # 注意：这里我们调用 infra 的接口。
                # 如果 infra 逻辑是简单的 'inject_event(mass)', 我们可能需要注入负质量来“清洗”势能
                # 或者依靠 infrastructure.py 中对 mud_detected 的特定处理 (如果已实现)
                # 为了通用性，我们模拟发送一个“低风险”信号
                # (Assuming infra simply accumulates events based on mass ~ risk)
                # 注入一个极小的 mass，或者如果可能，注入负值 (Physically impossible but cyber-possible)
                infra.flow_field.inject_event(mass=-2.0, velocity=10.0)
                logger.info(f"[Attack] {self.id} injecting FALSE SAFE signal at mud={real_mud}")

    def _build_semantic_packet(self, now):
        """
        [Attack Type 2: Physics Anomaly]
        Override broadcast packet to report physically impossible states.
        This is used to test the 'BayesianTrustFilter' defense mechanism.
        """
        packet = super()._build_semantic_packet(now)

        # 篡改数据：FDIA
        # 1. 谎报泥泞度 (诱骗全局规划)
        packet['mud_detected'] = 0.1

        # 2. 谎报速度 (测试物理一致性校验)
        # 在泥泞中 (real_mud > 0.6)，物理极限速度可能只有 2-3 m/s
        # 恶意节点谎称自己跑到了 20 m/s
        if self.env.get('mud_factor', 0.5) > 0.6:
            packet['vel'] = 20.0

        return packet