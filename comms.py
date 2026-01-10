import numpy as np
import math
import logging
from interfaces import ILinkLayer

# 配置日志
logger = logging.getLogger("CyberPhysical.Comms")


class SpectrumEnvironment:
    """
    [Cyber Security Layer]
    Global singleton representing the Electromagnetic Spectrum.
    Simulates Adversarial Jamming Attacks (DoS) in the edge network.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SpectrumEnvironment, cls).__new__(cls)
            cls._instance.jammers = []
            cls._instance.global_interference_floor = -100.0  # dBm
        return cls._instance

    def deploy_jammer(self, pos, radius, power_dbm):
        """Deploy a malicious jamming device."""
        self.jammers.append({'pos': np.array(pos), 'r': radius, 'p': power_dbm})
        logger.warning(f"[Security Alert] Jammer detected at {pos}, Radius={radius}m")

    def get_interference_at(self, pos):
        """Calculate total interference power (noise + jamming) at location."""
        total_mw = 10 ** (self.global_interference_floor / 10.0)

        for j in self.jammers:
            dist = np.linalg.norm(np.array(pos) - j['pos'])
            if dist < j['r']:
                # Jamming Power decays with distance squared
                jam_mw = 10 ** (j['p'] / 10.0) / (dist ** 2 + 1.0)
                total_mw += jam_mw

        return 10 * math.log10(total_mw)


class AgriculturalCommChannel(ILinkLayer):
    """
    [Physics-Aware Communication Model]
    Implements strong coupling between Environmental Physics (Mud) and Cyber Performance (Packet Loss).

    Physics-Cyber Coupling Equation:
    PL(d) = PL_0 + 10 * n(mud) * log(d) + Absorption(mud)
    """

    def __init__(self, tech: str, mud_factor: float, agent_pos: np.ndarray):
        self.tech = tech
        self.mud = mud_factor
        self.pos = agent_pos
        self.spectrum = SpectrumEnvironment()

        # 物理层参数设定
        if tech == "LoRa":
            self.freq = 433e6
            self.tx_power = 14  # dBm
            self.sensitivity = -130  # dBm
        elif tech == "WiFi":
            self.freq = 2.4e9
            self.tx_power = 20
            self.sensitivity = -85
        else:
            self.freq = 900e6
            self.tx_power = 10
            self.sensitivity = -110

        # [Coupling Logic 1] Path Loss Exponent heavily affected by environment
        # 泥浆环境会导致多径效应极其复杂，指数显著上升
        # Dry Soil: n=2.5,  Deep Mud: n=4.5
        self.n_path = 2.5 + 2.0 * np.clip(mud_factor, 0, 1)

    def update_position(self, new_pos):
        self.pos = np.array(new_pos)

    def transmit(self, dist: float, size: int, v: float) -> tuple:
        """
        Simulate packet transmission with Environmental & Adversarial effects.
        Returns: (success: bool, rssi: float, energy: float, latency: float, meta: dict)
        """
        dist = max(dist, 1.0)

        # 1. Physics-based Path Loss
        lambda_wave = 3e8 / self.freq
        pl_d0 = 20 * math.log10(4 * math.pi * 1.0 / lambda_wave)

        # [Coupling Logic 2] Mud Absorption Loss (Dielectric loss)
        # Wet mud absorbs RF energy, especially at high frequencies
        absorption_loss = 15.0 * self.mud * (self.freq / 1e9)

        path_loss = pl_d0 + 10 * self.n_path * math.log10(dist) + absorption_loss

        # 2. Stochastic Fading (Fast & Slow)
        shadowing = np.random.normal(0, 4.0)  # Obstacles
        # Doppler shift effect from velocity
        coherence_time = 3e8 / (self.freq * (abs(v) + 0.1))
        fading = np.random.exponential(1.0) if np.random.random() < 0.1 else 0.0  # Deep fade event

        # 3. Adversarial Interference (Cyber Security)
        # Calculate Signal-to-Interference-plus-Noise Ratio (SINR)
        noise_floor = self.spectrum.get_interference_at(self.pos)

        # Received Signal Strength
        rssi = self.tx_power - path_loss + shadowing - fading

        sinr = rssi - noise_floor

        # 4. Packet Error Rate (PER) Calculation
        # Waterfall curve approximation
        if self.tech == "LoRa":
            # LoRa is robust (CSS modulation)
            threshold = -20.0
            width = 2.0
        else:
            # WiFi is fragile (OFDM)
            threshold = 10.0
            width = 5.0

        # Sigmoid PER function
        per = 1.0 / (1.0 + np.exp((sinr - threshold) / width))

        # [Coupling Logic 3] Mechanical Vibration affecting Antenna contact
        # If moving fast on rough terrain (muddy), antenna wobbles causing burst errors
        if v > 5.0 and self.mud > 0.5:
            per = max(per, 0.2 * self.mud)

        is_success = np.random.random() > per

        # 5. Energy Cost (Joules)
        # Bad channel -> More retransmissions (MAC layer abstraction)
        retransmissions = 1 if is_success else 3
        datarate = 5000 if self.tech == "LoRa" else 6e6
        latency = (size * 8 / datarate) * retransmissions
        p_watt = (10 ** (self.tx_power / 10)) / 1000.0
        energy_cost = p_watt * latency

        debug_info = {
            'sinr': sinr,
            'jammed': noise_floor > -90.0,
            'absorption': absorption_loss
        }

        return is_success, rssi, energy_cost, latency, debug_info

    def get_diagnostics(self):
        return {'tech': self.tech, 'n_path': self.n_path}