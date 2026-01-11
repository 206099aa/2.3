import pandas as pd
import os
import logging
import time
from tqdm import tqdm
import networkx as nx  # [SCI] 新增引用：用于图搜索算法
from physics import DavisResistanceModel  # [SCI] 新增引用：用于物理能耗计算

# Import Core Modules
from config_loader import ConfigLoader
from map_core import GridMap
from vehicle import VehicleAgent

logging.basicConfig(level=logging.WARNING)  # 减少日志输出，提高速度


class HeadlessRunner:
    """
    [Experiment Automation]
    Runs batch simulations for statistical significance (Monte Carlo).
    """

    def __init__(self, config):
        self.cfg = config
        self.env = config['environment']
        self.grid = GridMap(config)

        # 收集道岔代理
        self.infra_agents = {
            nid: node.agent
            for nid, node in self.grid.nodes.items()
            if node.agent is not None
        }

        self.vehicles = []
        self._init_pop()
        self.logs = []

    def _init_pop(self):
        # 批量生成车辆
        # 示例：1个重型车，1个侦察车
        scenarios = [
            ('Heavy_Hauler', 'Start_1', 0.0),
            ('Fast_Scout', 'Start_2', 10.0)
        ]

        for i, (v_type, start_node, delay) in enumerate(scenarios):
            v = VehicleAgent(
                agent_id=f"V_{i}_{v_type}",
                vehicle_type_cfg=self.cfg['vehicle_types'][v_type],  # [修复] 正确传参
                env_config=self.env,
                start_node=start_node,
                map_graph=self.grid,
                infra_agents=self.infra_agents
            )
            self.vehicles.append(v)

        # Link V2V
        for veh in self.vehicles: veh.all_vehicles = self.vehicles

    def run_episode(self):
        t_max = self.cfg['simulation']['duration']
        dt = self.cfg['simulation']['dt']
        t = 0.0

        # 纯计算循环，无 GUI，速度极快
        while t < t_max:
            # 1. Update Infrastructure
            self.grid.update_infrastructure(dt, t)

            # 2. Update Vehicles
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
    利用全知视角（直接读取真实 Mud Field，无传感器噪声）计算理论最优解。
    用于生成 SCI 论文中的 "Optimality Gap" 基准线。
    """

    def __init__(self, config):
        self.cfg = config
        # 初始化地图（包含真实的泥泞场）
        self.grid = GridMap(config)
        self.davis = DavisResistanceModel()

        # 代价权重（必须与 router.py 中的 KinodynamicLinkEvaluator 保持一致以确保公平对比）
        self.alpha_t = 1.0  # Time weight
        self.beta_e = 0.1  # Energy weight

    def solve_theoretical_optimum(self, start_node, target_node, vehicle_type="Heavy_Hauler"):
        """
        运行全局 Dijkstra/A* 算法，寻找在当前环境下的理论物理最优路径。
        返回: (min_cost, optimal_time, optimal_energy, path_length)
        """
        # 1. 获取车辆物理参数
        v_spec = self.cfg['vehicle_types'][vehicle_type]
        mass = v_spec.get('mass_full', 5000.0)
        max_v = v_spec.get('max_speed', 12.0)

        # [关键] 强制使用全局配置的 Mud Factor，确保与 Physics 引擎一致
        global_mud = self.cfg['environment']['mud_factor']

        # 2. 定义 Oracle 代价函数 (God-Mode Cost Function)
        def oracle_weight(u, v, edge_attr):
            dist = edge_attr.get('length', 300.0)
            # [关键] 直接读取 Ground Truth 泥泞度，没有任何传感器噪声
            mud = global_mud

            # 物理极限速度估算 (与 router.py 逻辑一致，但数据是完美的)
            v_limit = max_v * (1.0 - 0.6 * mud)
            v_limit = max(1.0, v_limit)

            # A. 时间代价
            time_cost = dist / v_limit

            # B. 能耗代价 (Davis + Soil Mechanics)
            f_davis = self.davis.compute_resistance(mass, v_limit)
            f_soil = mass * 9.81 * (0.05 * mud)  # 简化的土壤阻力模型
            energy_cost = (f_davis + f_soil) * dist

            # 综合代价 J
            return self.alpha_t * time_cost + self.beta_e * energy_cost

        # 3. 运行全局最优寻路
        try:
            path = nx.dijkstra_path(
                self.grid.graph,
                start_node,
                target_node,
                weight=oracle_weight
            )

            # 4. 回溯计算该路径的各项指标
            total_time = 0.0
            total_energy = 0.0

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                data = self.grid.graph[u][v]

                # 重新计算物理量
                mud = global_mud
                dist = data.get('length', 300.0)
                v_act = max(1.0, max_v * (1.0 - 0.6 * mud))

                total_time += dist / v_act

                f_res = self.davis.compute_resistance(mass, v_act) + (mass * 9.81 * 0.05 * mud)
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
    # [SCI 核心配置] 定义扫描计划
    # 1. 泥泞度：从 0.1 到 0.9，每隔 0.1 测一次 -> 生成 Actuator Load 横向分布图
    # 2. 车辆类型：覆盖重载车和侦察车 -> 生成 Pareto 异构对比
    sweep_plan = {
        'environment.mud_factor': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # 如果你想跑得快一点，可以注释掉下面这行（只跑默认车辆）
        # 但为了 Pareto 图好看，建议保留
        'vehicle_types.Heavy_Hauler.pid.kp': [2000]
    }

    if not os.path.exists("config.yaml"):
        print("Error: config.yaml missing")
        exit()

    # --- Phase 1: Run Distributed Simulation Sweep ---
    all_results = []
    print("🚀 Phase 1: Running Distributed Simulation Sweep for SCI Analysis...")
    print("(This process simulates multiple episodes, please wait...)")

    # 生成配置矩阵
    configs = list(ConfigLoader.generate_sweep("config.yaml", sweep_plan))

    # 使用 tqdm 显示进度条
    for cfg in tqdm(configs):
        runner = HeadlessRunner(cfg)
        df = runner.run_episode()
        all_results.append(df)

    # 合并并保存
    if all_results:
        final_df = pd.concat(all_results)
        # 保存为 analysis-optimized.py 能识别的文件名格式
        final_df.to_csv("data/batch_results_sci.csv", index=False)
        print(f"✅ Distributed Data Saved ({len(final_df)} rows)")
    else:
        print("No results generated.")

    # --- Phase 2: Calculate Oracle Baselines ---
    print("\n🚀 Phase 2: Calculating Theoretical Upper Bound (Oracle)...")
    oracle_results = []

    # Load base config
    base_cfg = ConfigLoader.load("config.yaml")

    # 针对不同泥泞度计算理论最优解
    for mud in tqdm(sweep_plan['environment.mud_factor']):
        base_cfg['environment']['mud_factor'] = mud
        oracle = OracleRunner(base_cfg)

        # 假设典型任务: Start_1 -> N_3_3 (对角线任务，具体根据您的 map_core 拓扑调整)
        start_n = "Start_1"
        target_n = "N_3_3"

        res = oracle.solve_theoretical_optimum(start_n, target_n, "Heavy_Hauler")

        if res:
            oracle_results.append(res)

    if oracle_results:
        odf = pd.DataFrame(oracle_results)
        odf.to_csv("data/oracle_baseline.csv", index=False)
        print(f"✅ Oracle Data Saved: data/oracle_baseline.csv ({len(odf)} rows)")
    else:
        print("❌ Oracle failed to generate baselines.")