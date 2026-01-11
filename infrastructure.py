import numpy as np
import logging
import math
from collections import deque, defaultdict
from enum import Enum, auto
from dataclasses import dataclass

# 配置日志
logger = logging.getLogger("Edge.Infrastructure")


# =========================================================================
# [Layer 1] High-Fidelity Physics Kernel (ZD6 Switch Machine)
# -------------------------------------------------------------------------
# Electro-Thermal-Mechanical Coupling Model.
# =========================================================================

class SwitchState(Enum):
    LOCKED_NORMAL = auto()
    LOCKED_REVERSE = auto()
    UNLOCKING = auto()
    MOVING = auto()
    LOCKING = auto()
    STALLED = auto()  # Physical Failure
    COMPROMISED = auto()  # Cyber Failure (Hacked)


@dataclass
class ZD6PhysicalParams:
    """ZD6-D Switch Machine Specs (Railway Standard)."""
    R_armature_20C: float = 4.5
    L_armature: float = 0.05
    Ke: float = 0.85
    Kt: float = 0.85
    thermal_capacity: float = 500.0
    heat_dissipation: float = 2.5
    J_rotor: float = 0.02
    gear_ratio: float = 45.0
    screw_pitch: float = 0.01
    stroke: float = 0.16
    locking_force_peak: float = 2000.0


class ElectroThermalMotor:
    """Coupled Electro-Thermal Dynamics: L*di/dt + R(T)*i + Ke*w = V"""

    def __init__(self, params: ZD6PhysicalParams):
        self.p = params
        self.current = 0.0
        self.temperature = 25.0
        self.resistance = params.R_armature_20C

    def step_rk4(self, dt, voltage_in, omega_rotor):
        # 1. Thermal Update
        heat_gen = (self.current ** 2) * self.resistance
        heat_loss = self.p.heat_dissipation * (self.temperature - 25.0)
        self.temperature += (heat_gen - heat_loss) / self.p.thermal_capacity * dt
        # Temp coeff for Copper
        self.resistance = self.p.R_armature_20C * (1.0 + 0.004 * (self.temperature - 20.0))

        # 2. Electrical Update (RK4)
        def di_dt(i, v, w, r):
            return (v - i * r - self.p.Ke * w) / self.p.L_armature

        k1 = di_dt(self.current, voltage_in, omega_rotor, self.resistance)
        k2 = di_dt(self.current + 0.5 * dt * k1, voltage_in, omega_rotor, self.resistance)
        k3 = di_dt(self.current + 0.5 * dt * k2, voltage_in, omega_rotor, self.resistance)
        k4 = di_dt(self.current + dt * k3, voltage_in, omega_rotor, self.resistance)

        self.current += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.current = np.clip(self.current, -30.0, 30.0)
        return self.p.Kt * self.current


class ZD6Mechanism:
    """Mechanical Transmission with Environment-Coupled Friction."""

    def __init__(self, params: ZD6PhysicalParams, mud_factor: float):
        self.p = params
        self.mud = mud_factor
        self.omega = 0.0
        self.pos = 0.0
        self.efficiency = 0.8 * (1.0 - 0.3 * np.clip(mud_factor, 0, 1))

    def step(self, dt, motor_torque, external_force_N=0.0):
        k_linear = self.p.screw_pitch / (2 * np.pi * self.p.gear_ratio)
        v_linear = self.omega * k_linear

        # Hard Stops
        if (self.pos <= 0 and motor_torque < 0) or (self.pos >= self.p.stroke and motor_torque > 0):
            self.omega = 0.0
            return self.pos, self.omega

        # Stribeck Friction + Locking Force
        f_c = 300.0 + 500.0 * self.mud
        f_viscous = 1000.0 * (1.0 + 2.0 * self.mud) * v_linear
        f_fric = f_c * math.tanh(10.0 * v_linear) + f_viscous

        # Locking detent force at ends
        f_lock = 0.0
        if self.pos < 0.015 or self.pos > (self.p.stroke - 0.015):
            f_lock = self.p.locking_force_peak * np.sign(v_linear) if abs(v_linear) > 1e-4 else 0.0

        load_torque = (f_fric + f_lock + external_force_N) * k_linear / self.efficiency
        dw_dt = (motor_torque - load_torque - 0.05 * self.omega) / self.p.J_rotor

        self.omega += dw_dt * dt
        self.pos += (self.omega * k_linear) * dt
        self.pos = np.clip(self.pos, 0.0, self.p.stroke)
        return self.pos, self.omega


# =========================================================================
# [Layer 2] Secure Semantic Intelligence (Defensive Logic)
# -------------------------------------------------------------------------
# Implements Trust Filtering with Multi-Source Consensus (Problem 4 Fixed).
# =========================================================================

@dataclass
class SemanticOccupancy:
    owner_id: str
    arrival_mean: float
    duration_mean: float
    uncertainty_sigma: float
    last_update_ts: float


@dataclass
class EvidencePoint:
    value: float
    trust_weight: float
    timestamp: float


class BayesianTrustFilter:
    """
    [Cyber Security]
    Maintains a Trust Score for each vehicle ID.
    Filters out packets that violate physical laws OR consensus.
    """

    def __init__(self):
        self.occupancy_map = {}
        # Trust Score: 0.0 (Malicious) -> 1.0 (Trusted)
        self.trust_scores = defaultdict(lambda: 0.5)
        self.risk_level = 0.0

        # [Problem 4 Fix] Multi-Source Evidence Accumulator
        # Stores recent mud reports to detect semantic inconsistencies
        self.mud_evidence = deque(maxlen=20)

    def verify_and_update(self, packet, current_time):
        """
        Process packet only if it passes BOTH physics and semantic checks.
        """
        vid = packet['vid']
        trust = self.trust_scores[vid]

        # 1. Physics Plausibility Check (Hard Constraints)
        if not self._check_physics_constraints(packet):
            self.trust_scores[vid] *= 0.5  # Big penalty
            logger.warning(f"[Security] PHY-Violation from {vid}. Trust -> {self.trust_scores[vid]:.2f}")
            return False

        # 2. Semantic Consistency Check (Soft / Consensus Constraints)
        # Prevents "Lure Attacks" (Reporting safe road when it's dangerous)
        if not self._check_semantic_consistency(packet, current_time):
            self.trust_scores[vid] *= 0.6  # Penalty for lying
            logger.warning(f"[Security] SEM-Violation (Lure Attack?) from {vid}. Trust -> {self.trust_scores[vid]:.2f}")
            return False

        # If passed all checks:
        # Increase trust (Additive Increase)
        self.trust_scores[vid] = min(1.0, self.trust_scores[vid] + 0.05)

        # Record this evidence for future consensus
        if 'mud_detected' in packet:
            self.mud_evidence.append(EvidencePoint(
                value=packet['mud_detected'],
                trust_weight=self.trust_scores[vid],
                timestamp=current_time
            ))

        self._update_belief(packet, current_time)
        return True

    def _check_physics_constraints(self, packet):
        """[Defense 1] Verify physical possibility."""
        reported_vel = packet.get('vel', 0.0)
        # If mud is deep (e.g. inferred from own sensor or consensus), high speed is suspicious
        if abs(reported_vel) > 35.0:  # Hard limit for heavy machinery
            return False
        return True

    def _check_semantic_consistency(self, packet, now):
        """
        [Defense 2] Cross-Validation / Consensus.
        If this vehicle reports 'Clean' (0.1) but everyone else says 'Muddy' (0.8), reject it.
        """
        val = packet.get('mud_detected')
        if val is None: return True  # No semantic claim made

        # Filter out old evidence
        valid_evidence = [e for e in self.mud_evidence if (now - e.timestamp) < 30.0]

        if len(valid_evidence) < 3:
            return True  # Not enough data to judge, give benefit of doubt

        # Weighted Average of beliefs
        weighted_sum = sum(e.value * e.trust_weight for e in valid_evidence)
        total_weight = sum(e.trust_weight for e in valid_evidence) + 1e-6
        consensus_mud = weighted_sum / total_weight

        # Calculate Deviation
        deviation = abs(val - consensus_mud)

        # Dynamic threshold based on trust
        allowed_dev = 0.3 + 0.2 * self.trust_scores[packet['vid']]

        if deviation > allowed_dev:
            # Check if we are the outlier against a strong consensus
            if total_weight > 2.0:  # At least 2 full trusted cars worth of evidence
                logger.info(f"Consensus Check: {packet['vid']} says {val:.2f}, Consensus is {consensus_mud:.2f}")
                return False

        return True

    def _update_belief(self, packet, now):
        vid = packet['vid']
        aoi = max(0.0, now - packet.get('timestamp', now))
        trust_factor = 2.0 - self.trust_scores[vid]
        sigma = (packet.get('pos_uncertainty', 1.0) + 0.2 * aoi) * trust_factor

        self.occupancy_map[vid] = SemanticOccupancy(
            owner_id=vid,
            arrival_mean=packet.get('eta', now),
            duration_mean=packet.get('duration', 2.0),
            uncertainty_sigma=sigma,
            last_update_ts=now
        )
        self._prune(now)

    def _prune(self, now):
        dead = [k for k, v in self.occupancy_map.items() if (now - v.last_update_ts) > 30.0]
        for k in dead: del self.occupancy_map[k]
        self.risk_level = min(1.0, len(self.occupancy_map) * 0.2)

    def get_risk_prob(self, eta, dur):
        risk_accum = 0.0
        req_start, req_end = eta, eta + dur
        for occ in self.occupancy_map.values():
            buffer = 3.0 * occ.uncertainty_sigma
            if max(req_start, occ.arrival_mean - buffer) < min(req_end, occ.arrival_mean + occ.duration_mean + buffer):
                risk_accum += 1.0
        return risk_accum


# =========================================================================
# [Layer 3] Holographic Flow Field (Stigmergy)
# -------------------------------------------------------------------------
# Implements "Digital Pheromones" with Diffusion (Problem 1 Fixed).
# =========================================================================

class HolographicFlowField:
    """
    [Stigmergy]
    Nodes broadcast their 'Potential' (Pressure).
    Vehicles flow from High Potential (Congestion/Risk) to Low Potential.
    """

    def __init__(self, node_id):
        self.node_id = node_id
        self.local_pressure = 0.0  # Scalar Potential
        self.flow_vector = np.array([0.0, 0.0])
        self.turbulence = 0.0

        # [Problem 1 Fix] Field Parameters
        self.decay = 0.95  # Evaporation (Volatile memory)
        self.diffusion_rate = 0.5  # Diffusion coefficient (Spatial spread)

    def inject_event(self, mass=1.0, velocity=0.0):
        self.local_pressure += mass
        self.flow_vector += mass * velocity * np.array([1, 0])

    def set_repulsive_force(self):
        """Called when node fails. Sets huge pressure to repel traffic."""
        self.local_pressure = 1000.0
        self.turbulence = 1.0

    def compute_diffusion_influx(self, neighbor_potentials: list) -> float:
        """
        [Problem 1 Fix] Calculate Laplacian Diffusion Term.
        dP/dt_diffusion = D * sum(P_neighbor - P_self)
        """
        if not neighbor_potentials: return 0.0
        flux_sum = sum(p_j - self.local_pressure for p_j in neighbor_potentials)
        return self.diffusion_rate * flux_sum

    def apply_diffusion_update(self, delta_p):
        """Apply the calculated diffusion delta."""
        self.local_pressure += delta_p

    def step_decay(self, dt):
        """
        [Problem 1 Fix] Evaporation only. Diffusion is handled separately.
        """
        self.local_pressure *= (self.decay ** dt)
        self.flow_vector *= (self.decay ** dt)

        speed = np.linalg.norm(self.flow_vector)
        if self.local_pressure > 0.1:
            order = speed / (self.local_pressure * 10.0 + 1e-5)
            self.turbulence = 1.0 - np.clip(order, 0.0, 1.0)
        else:
            self.turbulence = 0.0

    def get_state(self):
        return {'potential': self.local_pressure, 'turbulence': self.turbulence}


# =========================================================================
# [Layer 4] Edge Switch Agent (Main Actor)
# -------------------------------------------------------------------------
# Integrates Physics, Security, and Flow Logic.
# =========================================================================

@dataclass
class TimeSlot:
    start: float
    end: float
    owner: str


class EdgeSwitchAgent:
    def __init__(self, node_id, env_config):
        self.id = node_id
        self.mud = env_config.get('mud_factor', 0.5)

        # 1. Physics
        self.params = ZD6PhysicalParams()
        self.motor = ElectroThermalMotor(self.params)
        self.mechanism = ZD6Mechanism(self.params, self.mud)
        self.state = SwitchState.LOCKED_NORMAL
        self.target_pos = 0.0
        self.health_index = 1.0
        self.stall_timer = 0.0

        # 2. Secure Intelligence
        self.trust_filter = BayesianTrustFilter()

        # 3. Flow Field
        self.flow_field = HolographicFlowField(node_id)

    def handle_semantic_packet(self, packet):
        """[V2I Interface]"""
        now = packet.get('timestamp', 0.0)

        # A. Trust Check (Cyber Defense: Physics + Semantic)
        if not self.trust_filter.verify_and_update(packet, now):
            # [Defense Active] Ignore the data payload from untrusted agent
            return {'status': 'REJECTED_UNTRUSTED', 'risk': 1.0}

        # B. Flow Injection (Only if trusted)
        self.flow_field.inject_event(mass=1.0, velocity=packet.get('vel', 0.0))

        # [NEW] Inject Semantic Mud Risk if reported
        # If trusted agent says mud is high, increase potential to warn others
        if packet.get('mud_detected', 0) > 0.6:
            self.flow_field.inject_event(mass=2.0, velocity=0.0)

        risk = self.trust_filter.get_risk_prob(packet.get('eta', now), packet.get('duration', 2.0))

        return {
            'status': 'ACK',
            'risk': risk,
            'potential': self.flow_field.local_pressure,
            'turbulence': self.flow_field.turbulence
        }

    def sync_diffusion(self, neighbor_potentials: list, dt: float):
        """
        [Problem 1 Fix] Phase 1 of Update: Spatial Diffusion.
        Receive potentials from neighbors and update local state.
        """
        influx = self.flow_field.compute_diffusion_influx(neighbor_potentials)
        # Apply integration: delta = rate * dt
        self.flow_field.apply_diffusion_update(influx * dt)

    def update(self, dt, time):
        """
        [Problem 1 Fix] Phase 2 of Update: Physics & Decay.
        """
        # 1. Physics Step
        if self.state == SwitchState.STALLED:
            self.flow_field.set_repulsive_force()
            return

        # Control Logic (PID)
        err = self.target_pos - self.mechanism.pos
        v_cmd = 48.0 * np.sign(err) if abs(err) > 0.01 else 0.0

        # Mechanical Sim
        torque = self.motor.step_rk4(dt, v_cmd, self.mechanism.omega)
        pos, vel = self.mechanism.step(dt, torque)
        self._update_fsm(pos, vel, self.motor.current, dt)

        # 2. Diffusion Step (Decay only, spatial diffusion happened in sync_diffusion)
        self.flow_field.step_decay(dt)

    def _update_fsm(self, pos, vel, current, dt):
        if abs(current) > 25.0 and abs(vel) < 0.1:
            self.stall_timer += dt
            if self.stall_timer > 1.0:
                self.state = SwitchState.STALLED
        else:
            self.stall_timer = 0.0

        if self.state == SwitchState.LOCKED_NORMAL and self.target_pos > 0.1:
            self.state = SwitchState.UNLOCKING

    def handle_hardware_signal(self, packet):
        if packet.get('type') == 'SWITCH_REQ':
            tgt = packet.get('target_state')
            self.target_pos = self.params.stroke if tgt == 'REVERSE' else 0.0