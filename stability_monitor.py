import numpy as np
import logging

# 配置日志
logger = logging.getLogger("Theory.Stability")


class LyapunovMonitor:
    """
    [Theoretical Validation Module]
    Implements a runtime Lyapunov function observer to empirically prove system stability.

    Lyapunov Candidate Function V(x):
    V(t) = V_tracking(t) + V_consensus(t) + V_energy(t)

    Where:
    - V_tracking: Sum of squared position/velocity errors (Control Performance).
    - V_consensus: Entropy of the holographic flow field (Distributed Agreement).
    - V_energy: Total generalized energy consumption (Efficiency).

    If dV/dt < 0, the system is asymptotically stable.
    """

    def __init__(self):
        self.history = []
        self.dt_accum = 0.0

        # 权重系数 (根据论文理论推导设定)
        self.alpha_track = 1.0
        self.beta_flow = 10.0
        self.gamma_energy = 0.001

    def step(self, t, vehicles, infra_nodes):
        """
        Compute V(t) at current time step.
        """
        # 1. Tracking Error Potential (V_tracking)
        # 衡量车辆是否偏离了其物理设定的目标状态
        v_track_sum = 0.0
        for v in vehicles:
            # 动能误差: 0.5 * m * (v - v_ref)^2
            # 这里简化为速度偏差的平方
            v_err = v.current_speed - getattr(v, 'target_speed_ref', 0.0)

            # [Physics Integration] 使用物理引擎中的实时质量
            # 注意：需确保 physics.py 已修复并包含 mass_total 属性
            mass = getattr(v.physics, 'mass_total', 5000.0)
            v_track_sum += 0.5 * mass * (v_err ** 2)

            # 势能误差: 距离目标的剩余距离 (L1 norm for robustness)
            if v.next_node_id:
                dist = np.linalg.norm(v.pos_2d - np.array(v.map.nodes[v.next_node_id].pos))
                v_track_sum += 100.0 * dist

        # 2. Flow Field Entropy (V_consensus)
        # 衡量分布式流场的混乱程度。如果协同良好，流场应趋于有序（低熵）。
        v_flow_sum = 0.0
        for nid, node in infra_nodes.items():
            if hasattr(node, 'agent') and node.agent:
                # 读取流场湍流度 (Turbulence)
                # Turbulence = 1 - Order_Parameter
                flow_state = node.agent.flow_field.get_state()
                turbulence = flow_state.get('turbulence', 0.0)
                potential = flow_state.get('potential', 0.0)

                # 加权: 只有在高势能(拥堵)区域，湍流才可怕
                v_flow_sum += turbulence * potential

        # 3. Energy Dissipation Potential (V_energy)
        # 系统总能耗作为一种广义势能，在优化过程中应被抑制
        # [Fix] Changed 'total_energy' to 'total' to match vehicle.py definition
        v_energy_sum = sum([v.energy.total for v in vehicles])

        # Total Lyapunov Function
        V_total = (self.alpha_track * v_track_sum +
                   self.beta_flow * v_flow_sum +
                   self.gamma_energy * v_energy_sum)

        # 记录数据
        self.history.append({
            'time': t,
            'V_total': V_total,
            'V_track': v_track_sum,
            'V_flow': v_flow_sum
        })

        # 实时稳定性检查 (Debugging)
        if len(self.history) > 100:
            dv = self.history[-1]['V_total'] - self.history[-2]['V_total']
            if dv > 1000.0:  # 允许小幅波动，但大幅增加意味着不稳定
                logger.warning(f"[Unstable] Lyapunov function increasing! dV={dv:.2f}")

        return V_total

    def get_convergence_rate(self):
        """
        Calculate exponential convergence rate lambda.
        V(t) ~ V(0) * e^(-lambda * t)
        """
        if len(self.history) < 100: return 0.0

        # 简单的对数回归
        try:
            times = np.array([h['time'] for h in self.history])
            values = np.array([h['V_total'] for h in self.history])

            # 避免 log(0)
            values = np.maximum(values, 1e-6)

            # Linear fit on log(V)
            slope, intercept = np.polyfit(times, np.log(values), 1)
            return -slope  # Lambda should be positive for stability
        except:
            return 0.0

    def export_data(self):
        import pandas as pd
        return pd.DataFrame(self.history)