import pandas as pd
import os
import logging
import time
from tqdm import tqdm
import networkx as nx
from physics import DavisResistanceModel

# Import Core Modules
from config_loader import ConfigLoader
from map_core import GridMap
from vehicle import VehicleAgent

logging.basicConfig(level=logging.WARNING)


class HeadlessRunner:
    """
    [Experiment Automation]
    Runs batch simulations for statistical significance.
    """

    def __init__(self, config):
        self.cfg = config
        self.env = config['environment']
        self.grid = GridMap(config)

        self.infra_agents = {
            nid: node.agent
            for nid, node in self.grid.nodes.items()
            if node.agent is not None
        }

        self.vehicles = []
        self._init_pop()
        self.logs = []

    def _init_pop(self):
        scenarios = [
            ('Heavy_Hauler', 'Start_1', 0.0),
            ('Fast_Scout', 'Start_2', 10.0)
        ]

        for i, (t_type, start_node, delay) in enumerate(scenarios):
            v = VehicleAgent(
                agent_id=f"V_{i}_{t_type}",
                train_type_name=t_type,
                global_config=self.cfg,
                start_node=start_node,
                map_graph=self.grid,
                infra_agents=self.infra_agents
            )
            self.vehicles.append(v)

        for veh in self.vehicles: veh.all_vehicles = self.vehicles

    def run_episode(self):
        t_max = self.cfg['simulation']['duration']
        dt = self.cfg['simulation']['dt']
        t = 0.0

        while t < t_max:
            self.grid.update_infrastructure(dt, t)
            for v in self.vehicles:
                log = v.step(dt, t)
                if log:
                    log.update({
                        'time': t,
                        'exp_id': self.cfg['meta']['experiment_id'],
                        'mud': self.env['mud_factor']
                    })
                    self.logs.append(log)
            t += dt

        return pd.DataFrame(self.logs)


class OracleRunner:
    """
    [Benchmark] God-Mode Oracle Solver.
    """

    def __init__(self, config):
        self.cfg = config
        self.grid = GridMap(config)
        self.davis = DavisResistanceModel()
        self.alpha_t = 1.0
        self.beta_e = 0.1

    def _estimate_convoy_mass(self, train_name):
        try:
            t_cfg = self.cfg['train_configurations'][train_name]
            specs = self.cfg['vehicle_specs']

            loco_mass = specs[t_cfg['locomotive']]['mass']

            w_spec = specs[t_cfg['wagon_type']]
            min_w, max_w = t_cfg['wagon_count']
            avg_count = (min_w + max_w) / 2.0

            tare = w_spec['tare_mass']
            max_load = w_spec['max_payload']
            min_fill = t_cfg['payload_distribution']['min_fill']
            max_fill = t_cfg['payload_distribution']['max_fill']
            avg_load = max_load * (min_fill + max_fill) / 2.0

            avg_wagon_mass = tare + avg_load

            total_mass = loco_mass + avg_count * avg_wagon_mass
            return total_mass
        except Exception as e:
            print(f"Oracle Mass Estimation Error: {e}")
            return 5000.0

    def solve_theoretical_optimum(self, start_node, target_node, train_type="Heavy_Hauler"):
        mass = self._estimate_convoy_mass(train_type)
        t_cfg = self.cfg['train_configurations'][train_type]
        max_v = self.cfg['vehicle_specs'][t_cfg['locomotive']]['max_speed']
        global_mud = self.cfg['environment']['mud_factor']

        def oracle_weight(u, v, edge_attr):
            dist = edge_attr.get('length', 300.0)
            mud = global_mud

            v_limit = max(1.0, max_v * (1.0 - 0.6 * mud))
            time_cost = dist / v_limit

            # [FIXED] Updated to use new physics API
            f_res = self.davis.compute(
                mass=mass,
                vel=v_limit,
                is_lead_unit=True,
                mud_factor=mud
            )

            energy_cost = f_res * dist

            return self.alpha_t * time_cost + self.beta_e * energy_cost

        try:
            path = nx.dijkstra_path(
                self.grid.graph,
                start_node,
                target_node,
                weight=oracle_weight
            )

            total_time = 0.0
            total_energy = 0.0

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                dist = self.grid.graph[u][v].get('length', 300.0)

                v_act = max(1.0, max_v * (1.0 - 0.6 * global_mud))
                total_time += dist / v_act

                # [FIXED] Updated API
                f_res = self.davis.compute(
                    mass=mass,
                    vel=v_act,
                    is_lead_unit=True,
                    mud_factor=global_mud
                )
                total_energy += f_res * dist

            return {
                'mud': global_mud,
                'oracle_time': total_time,
                'oracle_energy': total_energy,
                'oracle_path_len': len(path)
            }

        except nx.NetworkXNoPath:
            return None


if __name__ == "__main__":
    sweep_plan = {
        'environment.mud_factor': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'vehicle_specs.Loco_Class_A.pid.kp': [8000]
    }

    if not os.path.exists("config.yaml"):
        print("Error: config.yaml missing")
        exit()

    # --- Phase 1: Distributed Simulation ---
    all_results = []
    print("🚀 Phase 1: Running Distributed Simulation Sweep...")

    configs = list(ConfigLoader.generate_sweep("config.yaml", sweep_plan))

    for cfg in tqdm(configs):
        runner = HeadlessRunner(cfg)
        df = runner.run_episode()
        all_results.append(df)

    if all_results:
        final_df = pd.concat(all_results)
        final_df.to_csv("data/batch_results_sci.csv", index=False)
        print(f"✅ Distributed Data Saved ({len(final_df)} rows)")

    # --- Phase 2: Oracle Baselines ---
    print("\n🚀 Phase 2: Calculating Theoretical Upper Bound (Oracle)...")
    oracle_results = []

    base_cfg = ConfigLoader.load("config.yaml")

    for mud in tqdm(sweep_plan['environment.mud_factor']):
        base_cfg['environment']['mud_factor'] = mud
        oracle = OracleRunner(base_cfg)

        res = oracle.solve_theoretical_optimum("Start_1", "N_3_3", "Heavy_Hauler")

        if res:
            oracle_results.append(res)

    if oracle_results:
        odf = pd.DataFrame(oracle_results)
        odf.to_csv("data/oracle_baseline.csv", index=False)
        print(f"✅ Oracle Data Saved ({len(odf)} rows)")