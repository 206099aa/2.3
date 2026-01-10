import numpy as np
import math
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

# 配置日志
logger = logging.getLogger("HighFidelityPhysics")

GRAVITY = 9.81  # m/s^2


@dataclass
class EnvironmentState:
    mud_depth: float = 0.2
    rail_adhesion_factor: float = 1.0


class DavisResistanceModel:
    """
    [Physics Model] Davis Resistance Equation.
    R = A + B*v + C*v^2
    Used by BOTH the physical plant (for simulation) AND the cyber router (for estimation).
    """

    def __init__(self):
        # Coefficients for a typical flatbed railcar
        self.roll_coeff = 0.0015  # A term (Rolling resistance)
        self.flange_coeff = 0.0001  # B term (Flange friction)
        self.aero_drag_coeff = 0.3  # C term (Aerodynamic drag)

    def compute_resistance(self, mass_kg, velocity, frontal_area=2.0):
        """
        Calculate resistive force (Newtons).
        """
        v_abs = abs(velocity)

        # F_roll = mg * (C_roll)
        f_roll = mass_kg * GRAVITY * self.roll_coeff

        # F_mech = C_flange * v
        f_mech = mass_kg * self.flange_coeff * v_abs

        # F_aero = 0.5 * rho * Area * Cd * v^2
        f_aero = 0.5 * 1.225 * frontal_area * self.aero_drag_coeff * (v_abs ** 2)

        total_resistance = f_roll + f_mech + f_aero

        # Direction opposes motion
        return total_resistance * np.sign(velocity) if v_abs > 0.001 else 0.0


class PolachContactModel:
    """
    [Tribology] Polach Wheel-Rail Contact Model.
    Simulates complex adhesion reduction due to 'Third Body Layer' (Mud/Water).
    """

    def __init__(self):
        self.mu_0_dry = 0.55  # Maximum friction coefficient (Steel-Steel)

    def compute_adhesion_force(self, normal_force, v_wheel, v_vehicle, mud_factor):
        """
        Returns: (Adhesion Force, Effective Friction Coefficient)
        """
        epsilon = 1e-5
        v_ref = max(abs(v_vehicle), epsilon)

        # Creepage (Slip ratio)
        creepage = (v_wheel - v_vehicle) / v_ref

        # [Coupling Logic]
        # Mud creates a lubrication layer, significantly reducing mu
        # Exponential decay of friction with mud depth
        mu_available = self.mu_0_dry * np.exp(-2.0 * mud_factor)
        mu_available = max(mu_available, 0.05)  # Never zero, but very slippery

        if abs(creepage) < 1e-4:
            return 0.0, 0.0

        # Polach Curve Shape
        # Wet/Muddy rails have a flatter initial slope (K_creep)
        k_creep = 30.0 * (1.0 - 0.8 * mud_factor)

        tau = k_creep * creepage

        # Combine linear region and saturation
        # mu = mu_avail * ( (KA + (1-A)e^-B) ... simplified to tanh for stability)
        mu_eff = mu_available * np.tanh(abs(tau))

        # Add Stribeck effect for vibration simulation
        # At high slip, friction drops slightly (unstable region)
        if abs(creepage) > 0.05:
            mu_eff *= 0.9 + 0.1 * np.random.normal(0, 0.05 * mud_factor)

        return normal_force * mu_eff * np.sign(creepage), mu_eff


class DCMotorModel:
    """[Electrical] 400V DC Traction Motor."""

    def __init__(self, specs: Dict, max_current_limit: float = 400.0):
        self.R = specs.get('R', 0.05)
        self.L = specs.get('L', 0.01)
        self.Ke = specs.get('Ke', 1.0)  # Back-EMF constant
        self.Kt = specs.get('Kt', 1.0)  # Torque constant
        self.current = 0.0
        self.max_current_limit = max_current_limit

    def step_electrical(self, dt, voltage_in, omega_motor):
        """
        di/dt = (V - I*R - Ke*w) / L
        """
        back_emf = self.Ke * omega_motor
        di_dt = (voltage_in - self.current * self.R - back_emf) / self.L

        self.current += di_dt * dt
        self.current = np.clip(self.current, -self.max_current_limit, self.max_current_limit)

        return self.Kt * self.current, self.current


class RailVehicleMBDSystem:
    """
    [System Integration] Multi-Body Dynamics Plant.
    """

    def __init__(self, vehicle_config, env_config):
        self.cfg = vehicle_config
        self.mud_factor = env_config.get('mud_factor', 0.5)

        # Vehicle Physical Properties
        self.length = float(vehicle_config.get('length', 8.0))
        self.mass_total = float(vehicle_config.get('mass_full', 5000.0))

        # Mass Distribution (Locomotive vs Wagon)
        # Assuming 60% mass on drive wheels for traction
        self.mass_loco = self.mass_total * 0.6
        self.mass_wagon = self.mass_total * 0.4

        self.wheel_radius = 0.4
        self.gear_ratio = 8.0  # High torque for freight

        # System Voltage
        self.sys_voltage = 400.0

        # Motor Specs
        self.davis_model = DavisResistanceModel()
        self.contact_model = PolachContactModel()

        # Tuned Motor for Heavy Load
        self.motor = DCMotorModel(
            {'R': 0.05, 'L': 0.01, 'Ke': 1.5, 'Kt': 1.5},
            max_current_limit=600.0
        )

        # State Vector: [x_loco, v_loco, x_wag, v_wag]
        self.dof = 4
        self.state = np.zeros(self.dof)
        self.last_mu_eff = 0.0

        # Soft Coupler
        self.coupler_gap = 0.05
        self.k_coupler = 5.0e4
        self.c_coupler = 5.0e3

        self._init_position()

    def _init_position(self, spacing=3.0):
        # Initialize loco and wagon with spacing
        self.state[0] = 0.0
        self.state[2] = -spacing

    def _calculate_coupler_force(self):
        x_loco, v_loco = self.state[0], self.state[1]
        x_wag, v_wag = self.state[2], self.state[3]

        dist = x_loco - x_wag
        nominal = 3.0  # Nominal length

        dx = dist - nominal
        dv = v_loco - v_wag

        force = 0.0
        # Non-linear spring with deadband (slack)
        if abs(dx) > self.coupler_gap:
            eff_dx = dx - np.sign(dx) * self.coupler_gap
            force = self.k_coupler * eff_dx + self.c_coupler * dv

        return force

    def get_derivatives(self, t, state, motor_torque, bus_voltage):
        v_loco = state[1]
        v_wag = state[3]

        # 1. Drive Force
        f_drive_mech = (motor_torque * self.gear_ratio) / self.wheel_radius

        # 2. Wheel Slip / Traction Limit
        # Estimate wheel speed based on back-EMF logic or simply motor speed
        # Here we simplify: w_wheel is related to v_loco but slips if torque is high

        # Effective wheel velocity (driven)
        # If torque is huge and resistance is high, wheel spins up
        # We model this via the contact model directly
        # For MBD, we usually need a separate state for wheel omega.
        # Simplified here: assume w_wheel matches v_loco unless slipping

        # To enable slip, we'd need J_wheel * dw/dt = T_motor - T_friction
        # Here we use a Quasi-Static assumption for slip to keep it stable:
        # Slip is proportional to Torque / NormalForce

        slip_ratio = (f_drive_mech / (self.mass_loco * GRAVITY)) * 0.1
        v_wheel = v_loco * (1.0 + slip_ratio)

        f_traction_limit, mu = self.contact_model.compute_adhesion_force(
            self.mass_loco * GRAVITY, v_wheel, v_loco, self.mud_factor
        )
        self.last_mu_eff = mu

        # Clamp drive force to adhesion limit
        f_net_drive = np.clip(f_drive_mech, -abs(f_traction_limit), abs(f_traction_limit))

        # 3. Resistances
        f_res_loco = self.davis_model.compute_resistance(self.mass_loco, v_loco, 2.0)
        f_res_wag = self.davis_model.compute_resistance(self.mass_wagon, v_wag, 2.0)

        # Soil Drag (Sinkage)
        f_soil_loco = self.mass_loco * GRAVITY * 0.1 * self.mud_factor * np.sign(v_loco)

        # 4. Coupler
        f_c = self._calculate_coupler_force()

        # 5. EOM
        # Loco: ma = F_drive - F_res - F_soil - F_coupler
        a_loco = (f_net_drive - f_res_loco - f_soil_loco - f_c) / self.mass_loco

        # Wagon: ma = F_coupler - F_res - F_soil
        a_wag = (f_c - f_res_wag) / self.mass_wagon

        return np.array([v_loco, a_loco, v_wag, a_wag])

    def step_rk4(self, dt, control_u):
        """
        RK4 Integration. control_u is Voltage (-48 to 48 or -400 to 400).
        """
        # Scale input to System Voltage
        bus_voltage = control_u * (self.sys_voltage / 48.0)
        bus_voltage = np.clip(bus_voltage, -self.sys_voltage, self.sys_voltage)

        # Sub-stepping for electrical stability
        n_sub = 10
        dt_s = dt / n_sub

        y = self.state.copy()

        for _ in range(n_sub):
            # Update Motor Current
            omega_wheel = y[1] / self.wheel_radius
            omega_motor = omega_wheel * self.gear_ratio
            torque, _ = self.motor.step_electrical(dt_s, bus_voltage, omega_motor)

            # Update MBD
            k1 = self.get_derivatives(0, y, torque, bus_voltage)
            k2 = self.get_derivatives(0, y + 0.5 * dt_s * k1, torque, bus_voltage)
            k3 = self.get_derivatives(0, y + 0.5 * dt_s * k2, torque, bus_voltage)
            k4 = self.get_derivatives(0, y + dt_s * k3, torque, bus_voltage)

            y += (dt_s / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.state = y

        return {
            'loco_vel': self.state[1],
            'motor_current': self.motor.current,
            'coupler_force_1': self._calculate_coupler_force(),
            'mu_effective': self.last_mu_eff
        }