import logging
import time
import os
import pandas as pd
from datetime import datetime

# Import Core Modules
from config_loader import ConfigLoader
from map_core import GridMap
# [SCI] 显式导入 MaliciousVehicle 以启用攻击
from vehicle import VehicleAgent, MaliciousVehicle

# [SCI Integration] New Modules
from stability_monitor import LyapunovMonitor
from comms import SpectrumEnvironment
from visualization import SimVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger("MainLoop")


class DecentralizedSimulation:
    """
    [Experiment Orchestrator]
    Physics-Aware Distributed Edge Control Simulation.
    """

    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config {config_path} not found")

        self.cfg = ConfigLoader.load(config_path)

        # 1. Initialize Spectrum (Cyber Environment)
        self.spectrum = SpectrumEnvironment()
        self.spectrum.deploy_jammer(pos=[300, 300], radius=100.0, power_dbm=20.0)

        # 2. Initialize Heterogeneous Map & Edge Infrastructure
        logger.info("Initializing Map with Unstructured Mud Fields...")
        self.map = GridMap(self.cfg)

        self.infra_agents = {
            nid: node.agent
            for nid, node in self.map.nodes.items()
            if node.agent is not None
        }
        logger.info(f"Deployed {len(self.infra_agents)} Distributed Edge Nodes.")

        # 3. Initialize Theoretical Monitor
        self.monitor = LyapunovMonitor()

        # 4. Initialize Heterogeneous Fleet
        logger.info("Deploying Heterogeneous Fleet...")
        self.vehicles = []
        self._deploy_fleet()

        # 5. Data Logging & Vis
        self.logs = []
        self.viz = SimVisualizer(self.map, self.vehicles, self.cfg)

    def _deploy_fleet(self):
        """
        Deploy Mixed Platoon: Scouts (Fast/Light) + Haulers (Heavy/Slow).
        [Fixed] Updated instantiation to match new VehicleAgent signature.
        """
        # Configuration for 4 vehicles
        # ID, Train_Type_Name, Start_Node, Is_Malicious_Class
        deployment_plan = [
            # Group 1: Normal
            ("Scout_Alpha", "Fast_Scout", "Start_1", False),
            ("Hauler_One", "Heavy_Hauler", "Start_1", False),

            # Group 2: Under Attack
            ("Malicious_Attacker", "Fast_Scout", "Start_2", True),
            ("Hauler_Two", "Heavy_Hauler", "Start_2", False)
        ]

        for v_id, t_type, start_node, is_malicious in deployment_plan:
            # Select Agent Class
            AgentClass = MaliciousVehicle if is_malicious else VehicleAgent

            # [Fix] Pass 't_type' (string) and 'self.cfg' (full config)
            agent = AgentClass(
                agent_id=v_id,
                train_type_name=t_type,  # <--- Corrected Argument
                global_config=self.cfg,  # <--- Corrected Argument
                start_node=start_node,
                map_graph=self.map,
                infra_agents=self.infra_agents
            )
            self.vehicles.append(agent)

            role = "MALICIOUS" if is_malicious else t_type
            logger.info(f" -> Deployed {v_id} ({role}) at {start_node}")

    def step(self, t, dt):
        # 1. Infrastructure Update
        self.map.update_infrastructure(dt, t)

        # 2. Vehicle Agents Update
        step_data = []
        for v in self.vehicles:
            telemetry = v.step(dt, t)
            if telemetry:
                log_entry = telemetry.copy()
                log_entry.update({
                    'time': t,
                    'mud_global': self.cfg['environment']['mud_factor'],
                    'exp_id': self.cfg['meta']['experiment_id']
                })
                step_data.append(log_entry)

        self.logs.extend(step_data)

        # 3. Theoretical Validation
        V_t = self.monitor.step(t, self.vehicles, self.infra_agents)

        if int(t) % 10 == 0 and abs(t - int(t)) < dt / 2:
            print(f"T={t:04.0f}s | Lyapunov V={V_t:.2f} | Active Vehicles={len(self.vehicles)}", end='\r')

    def run(self):
        duration = self.cfg['simulation']['duration']
        dt = self.cfg['simulation']['dt']

        logger.info(f"Starting Simulation (T={duration}s)...")

        def sim_generator():
            t = 0.0
            step_count = 0
            RENDER_SKIP = 20

            while t < duration:
                self.step(t, dt)
                step_count += 1
                if step_count % RENDER_SKIP == 0:
                    yield t, self.vehicles
                t += dt

        try:
            if self.cfg['simulation']['visualization']:
                self.viz.start(sim_generator)
            else:
                for _ in sim_generator(): pass
        except KeyboardInterrupt:
            logger.warning("Simulation stopped by user.")
        finally:
            self._save_results()

    def _save_results(self):
        if not self.logs: return
        if not os.path.exists('data'): os.makedirs('data')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        df = pd.DataFrame(self.logs)
        filename = f"data/SCI_Exp_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"\nExperimental Data Saved: {filename}")

        df_stability = self.monitor.export_data()
        file_stab = f"data/SCI_Stability_{timestamp}.csv"
        df_stability.to_csv(file_stab, index=False)
        logger.info(f"Stability Proof Saved: {file_stab}")


if __name__ == "__main__":
    sim = DecentralizedSimulation()
    sim.run()