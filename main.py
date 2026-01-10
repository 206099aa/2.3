import logging
import time
import os
import pandas as pd
from datetime import datetime

# Import Core Modules
from config_loader import ConfigLoader
from map_core import GridMap
from vehicle import VehicleAgent

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

    Key Features:
    1. No Central Scheduler (Fully Distributed).
    2. Heterogeneous Agents (Scouts + Haulers).
    3. Adversarial Environment (Jamming + Mud).
    4. Theoretical Monitoring (Lyapunov).
    """

    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config {config_path} not found")

        self.cfg = ConfigLoader.load(config_path)
        self.env_cfg = self.cfg['environment']

        # 1. Initialize Spectrum (Cyber Environment)
        self.spectrum = SpectrumEnvironment()
        # [Adversarial] Deploy a Jammer at a critical junction
        self.spectrum.deploy_jammer(pos=[300, 300], radius=100.0, power_dbm=20.0)

        # 2. Initialize Heterogeneous Map & Edge Infrastructure
        logger.info("Initializing Map with Unstructured Mud Fields...")
        self.map = GridMap(self.cfg)

        # Collect Edge Agents
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
        """
        # Configuration for 4 vehicles
        deployment_plan = [
            # ID, Type, Start Node
            ("Scout_Alpha", "Fast_Scout", "Start_1"),
            ("Hauler_One", "Heavy_Hauler", "Start_1"),  # Follows Alpha
            ("Scout_Beta", "Fast_Scout", "Start_2"),
            ("Hauler_Two", "Heavy_Hauler", "Start_2")
        ]

        for v_id, v_type, start_node in deployment_plan:
            agent = VehicleAgent(
                agent_id=v_id,
                vehicle_type_cfg=self.cfg['vehicle_types'][v_type],
                env_config=self.env_cfg,
                start_node=start_node,
                map_graph=self.map,
                infra_agents=self.infra_agents
            )
            self.vehicles.append(agent)
            logger.info(f" -> Deployed {v_id} ({v_type}) at {start_node}")

    def step(self, t, dt):
        """
        [Parallel Execution Emulation]
        Simulates one time-step of the Distributed System.
        Strictly NO central control logic here.
        """
        # 1. Infrastructure Update (Edge Computing & Physics)
        self.map.update_infrastructure(dt, t)

        # 2. Vehicle Agents Update (Sense -> Plan -> Act)
        step_data = []
        for v in self.vehicles:
            # Agents run autonomously
            telemetry = v.step(dt, t)

            # [Data Collection]
            if telemetry:
                log_entry = telemetry.copy()
                log_entry.update({
                    'time': t,
                    'mud_global': self.env_cfg['mud_factor'],  # Ground truth for analysis
                    'exp_id': self.cfg['meta']['experiment_id']
                })
                step_data.append(log_entry)

        self.logs.extend(step_data)

        # 3. Theoretical Validation (Observer, not Controller)
        V_t = self.monitor.step(t, self.vehicles, self.infra_agents)

        # [Console Heartbeat]
        if int(t) % 10 == 0 and abs(t - int(t)) < dt / 2:
            print(f"T={t:04.0f}s | Lyapunov V={V_t:.2f} | Active Vehicles={len(self.vehicles)}", end='\r')

    def run(self):
        duration = self.cfg['simulation']['duration']
        dt = self.cfg['simulation']['dt']

        logger.info(f"Starting Simulation (T={duration}s)...")

        # Generator for Visualization
        def sim_generator():
            t = 0.0
            step_count = 0
            RENDER_SKIP = 20  # 50Hz physics -> 2.5Hz render

            while t < duration:
                self.step(t, dt)

                step_count += 1
                if step_count % RENDER_SKIP == 0:
                    yield t, self.vehicles

                t += dt

        # Start Loop
        try:
            # Check if headless or visual
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

        # 1. Save Experimental Data
        df = pd.DataFrame(self.logs)
        filename = f"data/SCI_Exp_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"\nExperimental Data Saved: {filename}")

        # 2. Save Lyapunov Stability Proof
        df_stability = self.monitor.export_data()
        file_stab = f"data/SCI_Stability_{timestamp}.csv"
        df_stability.to_csv(file_stab, index=False)
        logger.info(f"Stability Proof Saved: {file_stab}")


if __name__ == "__main__":
    sim = DecentralizedSimulation()
    sim.run()