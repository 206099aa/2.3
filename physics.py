import numpy as np
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# 配置日志
logger = logging.getLogger("Physics.MBD")

GRAVITY = 9.81  # m/s^2


# =========================================================================
# [Sub-Model] Tribology & Resistance
# =========================================================================

class PolachContactModel:
    """
    [Tribology] Polach Wheel-Rail Contact Model.
    Simulates complex adhesion reduction due to 'Third Body Layer' (Mud/Water).
    """

    def __init__(self):
        self.mu_0_dry = 0.55  # Maximum friction coefficient (Steel-Steel)

    def compute_adhesion_limit(self, normal_force, v_wheel, v_vehicle, mud_factor):
        """
        Returns: (Max Available Traction Force, Effective Friction Coefficient)
        """
        epsilon = 1e-5
        v_ref = max(abs(v_vehicle), epsilon)

        # Creepage (Slip ratio)
        # Simplified: Assume control system maintains optimal slip,
        # but environment reduces the peak of the curve.

        # [Coupling Logic]
        # Mud creates a lubrication layer, significantly reducing mu
        # Exponential decay of friction with mud depth
        mu_available = self.mu_0_dry * np.exp(-2.5 * mud_factor)
        mu_available = max(mu_available, 0.05)  # Never zero, but very slippery (0.05 ~ Ice)

        # Stochastic Stribeck effect (Vibration on rough terrain)
        if abs(v_vehicle) > 0.1:
            noise = np.random.normal(0, 0.02 * mud_factor)
            mu_available = np.clip(mu_available + noise, 0.01, 0.6)

        return normal_force * mu_available, mu_available


class DavisResistanceModel:
    """
    [Physics Model] Generalized Davis Equation.
    R = A + B*v + C*v^2 + F_grade + F_curve + F_soil
    """

    def __init__(self):
        self.roll_coeff = 0.0015  # A term
        self.flange_coeff = 0.0001  # B term
        self.aero_drag_coeff = 0.3  # C term (Frontal)
        self.skin_friction = 0.05  # C term (Side/Skin for wagons)

    def compute(self, mass, vel, is_lead_unit, area=10.0, mud_factor=0.0):
        v_abs = abs(vel)

        # 1. Rolling Resistance (F = mg * C_roll)
        f_roll = mass * GRAVITY * self.roll_coeff

        # 2. Mechanical/Flange Friction (F = C_flange * v)
        f_mech = mass * self.flange_coeff * v_abs

        # 3. Aerodynamic Drag (F = 0.5 * rho * A * Cd * v^2)
        # Lead unit takes full frontal drag, trailing units take skin friction
        rho = 1.225
        cd = self.aero_drag_coeff if is_lead_unit else self.skin_friction
        f_aero = 0.5 * rho * area * cd * (v_abs ** 2)

        # 4. Environmental Resistance (Mud Sinkage / Drag)
        # Empirical: Resistance increases linearly with mud depth and mass
        f_mud = mass * GRAVITY * (0.02 * mud_factor)
        if mud_factor > 0.5:
            # Non-linear penalty for deep mud (bogging down)
            f_mud *= (1.0 + (mud_factor - 0.5) * 2.0)

        total = f_roll + f_mech + f_aero + f_mud
        return total * np.sign(vel) if v_abs > 0.001 else 0.0


class CouplerModel:
    """
    [Dynamics] Non-linear Spring-Damper with Slack (Gap).
    Models the connection between railcars.
    """

    def __init__(self, stiffness=2.0e6, damping=5.0e4, gap=0.05):
        self.k = stiffness
        self.c = damping
        self.gap = gap  # m (Dead zone)
        self.nominal_dist = 1.0  # m (Ideal distance between mass centers offset)

    def compute_force(self, dx, dv):
        """
        dx: Relative displacement (x_front - x_rear) - nominal
        dv: Relative velocity (v_front - v_rear)
        """
        # Dead zone (Slack) logic
        # Force is zero if within the gap
        force = 0.0

        if dx > self.gap / 2:
            # Tension (Pulling)
            eff_x = dx - self.gap / 2
            force = self.k * eff_x + self.c * dv
        elif dx < -self.gap / 2:
            # Compression (Pushing/Buffing)
            eff_x = dx + self.gap / 2
            force = self.k * eff_x + self.c * dv

        return force


# =========================================================================
# [Atomic Units] Locomotive & Wagon
# =========================================================================

class RollingStock:
    """Base class for any rail vehicle unit."""

    def __init__(self, mass, length, area=8.0):
        self.mass = mass
        self.length = length
        self.area = area
        self.pos = 0.0
        self.vel = 0.0
        self.acc = 0.0

    def get_mass(self):
        return self.mass


class Locomotive(RollingStock):
    def __init__(self, spec: Dict):
        super().__init__(spec['mass'], spec['length'])
        self.max_power = spec['max_power']
        self.max_force = spec.get('max_force', 600000.0)  # N
        self.adhesion_weight = spec.get('adhesion_weight', self.mass)

        # Sub-systems
        self.motor = DCMotorModel({'R': 0.05, 'L': 0.01, 'Ke': 1.5, 'Kt': 1.5})
        self.contact = PolachContactModel()

    def get_tractive_effort(self, throttle_signal, velocity, mud_factor, dt):
        """
        Calculate net tractive force considering Motor limits and Adhesion limits.
        throttle: -1.0 (Max Brake) to 1.0 (Max Power)
        """
        # 1. Electrical Torque Generation
        # Simplified: Power = Force * Velocity -> Force = Power / Velocity
        target_power = abs(throttle_signal) * self.max_power

        if abs(velocity) < 0.1:
            f_motor = self.max_force * abs(throttle_signal)  # Low speed const force
        else:
            f_motor = min(self.max_force, target_power / abs(velocity))

        f_motor *= np.sign(throttle_signal)

        # 2. Adhesion Limit (Physics Check)
        normal_force = self.adhesion_weight * GRAVITY
        # Assume wheel speed ~ vehicle speed for simple adhesion check
        # (Detailed slip dynamics omitted for simulation speed, using quasi-static limit)
        f_limit, mu_eff = self.contact.compute_adhesion_limit(normal_force, velocity, velocity, mud_factor)

        # 3. Clamp Force
        f_final = np.clip(f_motor, -f_limit, f_limit)

        return f_final, mu_eff, self.motor.current  # Approx current


class Wagon(RollingStock):
    def __init__(self, spec: Dict, payload_ratio: float = 1.0):
        # Calculate dynamic mass based on random payload
        tare = spec['tare_mass']
        max_load = spec['max_payload']
        actual_load = max_load * payload_ratio

        total_mass = tare + actual_load
        super().__init__(total_mass, spec['length'])
        self.payload = actual_load


class DCMotorModel:
    """Simple First-Order DC Motor Lag"""

    def __init__(self, specs):
        self.current = 0.0
    # Placeholder for more complex electrical dynamics if needed


# =========================================================================
# [System Integrator] Train Convoy
# =========================================================================

class TrainConvoy:
    """
    [MBD Core] Multi-Body Dynamics System for a Train Convoy.
    Solves coupled differential equations for N connected bodies.
    """

    def __init__(self, train_config_name: str, global_config: Dict):
        self.cfg = global_config
        self.units: List[RollingStock] = []
        self.couplers: List[CouplerModel] = []
        self.davis = DavisResistanceModel()

        self._build_train(train_config_name)

        # State Vector: [x0, v0, x1, v1, ..., xn, vn]
        self.dof = len(self.units) * 2
        self.state = np.zeros(self.dof)

        # Init positions (lined up)
        current_x = 0.0
        for i, unit in enumerate(self.units):
            self.state[2 * i] = current_x
            self.state[2 * i + 1] = 0.0  # Initial velocity
            if i < len(self.units) - 1:
                # Place next unit behind (assuming standard coupler length)
                # Distance = half_len_i + coupler_dist + half_len_next
                dist = unit.length / 2 + 1.0 + self.units[i + 1].length / 2
                current_x -= dist

        logger.info(
            f"Initialized Train {train_config_name} with {len(self.units)} units. Total Mass: {sum(u.mass for u in self.units) / 1000:.1f}t")

    def _build_train(self, config_name):
        """Construct the convoy from config specs."""
        t_cfg = self.cfg['train_configurations'][config_name]
        specs = self.cfg['vehicle_specs']

        # 1. Add Locomotive
        loco_type = t_cfg['locomotive']
        self.units.append(Locomotive(specs[loco_type]))

        # 2. Add Wagons
        wagon_type = t_cfg['wagon_type']
        # Random count
        min_w, max_w = t_cfg['wagon_count']
        count = np.random.randint(min_w, max_w + 1)

        # Payload distribution
        p_min = t_cfg['payload_distribution']['min_fill']
        p_max = t_cfg['payload_distribution']['max_fill']

        for _ in range(count):
            # Random payload for heterogeneity
            p_ratio = np.random.uniform(p_min, p_max)
            self.units.append(Wagon(specs[wagon_type], p_ratio))

            # Add coupler connecting previous unit to this one
            self.couplers.append(CouplerModel())

    def get_derivatives(self, t, state, control_u, mud_factor):
        """
        Compute dx/dt for the whole system.
        control_u: Throttle input (-1.0 to 1.0)
        """
        dydt = np.zeros_like(state)

        # --- 1. Compute Internal Coupler Forces ---
        coupler_forces = []  # Forces acting on the gap i

        for i, coupler in enumerate(self.couplers):
            # Unit i (Front) vs Unit i+1 (Rear)
            idx_front = i
            idx_rear = i + 1

            x_f, v_f = state[2 * idx_front], state[2 * idx_front + 1]
            x_r, v_r = state[2 * idx_rear], state[2 * idx_rear + 1]

            # Geometric distance
            # nominal_center_dist = half_len_f + 1.0 + half_len_r
            nom_dist = self.units[idx_front].length / 2 + 1.0 + self.units[idx_rear].length / 2

            dx = (x_f - x_r) - nom_dist
            dv = v_f - v_r

            f_c = coupler.compute_force(dx, dv)
            coupler_forces.append(f_c)

        # --- 2. Solve EOM for each unit ---
        # Loco physics data for telemetry
        loco_traction = 0.0
        loco_mu = 0.0

        for i, unit in enumerate(self.units):
            # State indices
            idx_x = 2 * i
            idx_v = 2 * i + 1
            v_curr = state[idx_v]

            # A. Environmental Resistance
            f_res = self.davis.compute(unit.mass, v_curr, is_lead_unit=(i == 0), mud_factor=mud_factor)

            # B. Traction/Braking (Only for Loco)
            f_trac = 0.0
            if isinstance(unit, Locomotive):
                f_trac, mu, _ = unit.get_tractive_effort(control_u, v_curr, mud_factor, 0)
                loco_traction = f_trac
                loco_mu = mu
            else:
                # Wagon Brakes (Simplified: 50% of loco braking command)
                if control_u < 0:
                    brake_force = abs(control_u) * 0.5 * unit.mass * 9.81 * 0.1  # Friction brake
                    f_trac = -brake_force * np.sign(v_curr) if abs(v_curr) > 0.1 else 0.0

            # C. Coupler Forces
            # Unit i is pulled BACK by coupler i (if exists)
            # Unit i is pushed FRONT by coupler i-1 (if exists)
            f_couple_net = 0.0

            # Rear coupler (Pulling back)
            if i < len(self.couplers):
                f_couple_net -= coupler_forces[i]

            # Front coupler (Pulling forward)
            if i > 0:
                f_couple_net += coupler_forces[i - 1]

            # D. Net Force & Acceleration
            # F = ma
            f_net = f_trac - f_res + f_couple_net
            acc = f_net / unit.mass

            dydt[idx_x] = v_curr
            dydt[idx_v] = acc

        return dydt, loco_traction, loco_mu, (coupler_forces[0] if coupler_forces else 0.0)

    def step_rk4(self, dt, control_u, mud_factor):
        """
        Runge-Kutta 4th Order Integration.
        """
        y = self.state.copy()

        # K1
        k1, tr, mu, cf = self.get_derivatives(0, y, control_u, mud_factor)

        # K2
        k2, _, _, _ = self.get_derivatives(0, y + 0.5 * dt * k1, control_u, mud_factor)

        # K3
        k3, _, _, _ = self.get_derivatives(0, y + 0.5 * dt * k2, control_u, mud_factor)

        # K4
        k4, _, _, _ = self.get_derivatives(0, y + dt * k3, control_u, mud_factor)

        self.state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Return telemetry of the Locomotive (Lead Unit)
        return {
            'loco_vel': self.state[1],
            'motor_current': tr / 400.0,  # Dummy scaling for current
            'coupler_force_1': cf,
            'mu_effective': mu,
            'total_mass': sum([u.mass for u in self.units])
        }

    def _init_position(self, spacing):
        # Already handled in __init__
        pass