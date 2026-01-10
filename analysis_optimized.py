import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import glob
from scipy.signal import welch
from scipy.stats import pearsonr

# 配置 SCI 绘图风格 (IEEE Standard)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'lines.linewidth': 1.5,
    'figure.autolayout': True
})

DATA_DIR = "data"


class DeepSCIAnalyzer:
    """
    [Paper Logic]
    Generates 4 Key Figures for "Physics-Aware Distributed Edge Control":
    1. Theoretical Proof: Lyapunov Stability Convergence (Fig 3).
    2. Resilience: System behavior under Cyber-Attack (Fig 4).
    3. Performance: Pareto Efficiency Frontier (Fig 5).
    4. Physics Validation: Environmental Coupling (Fig 6).
    """

    def __init__(self):
        self.exp_df = None
        self.stab_df = None
        self._load_latest_data()

    def _load_latest_data(self):
        # 1. Load Experiment Data (SCI_Exp_*.csv)
        exp_files = glob.glob(os.path.join(DATA_DIR, "SCI_Exp_*.csv"))
        if exp_files:
            latest_exp = max(exp_files, key=os.path.getmtime)
            print(f"📂 Loaded Experiment Data: {latest_exp}")
            self.exp_df = pd.read_csv(latest_exp)
        else:
            print("⚠️ No Experiment Data found!")

        # 2. Load Stability Data (SCI_Stability_*.csv)
        stab_files = glob.glob(os.path.join(DATA_DIR, "SCI_Stability_*.csv"))
        if stab_files:
            latest_stab = max(stab_files, key=os.path.getmtime)
            print(f"📂 Loaded Stability Proof: {latest_stab}")
            self.stab_df = pd.read_csv(latest_stab)
        else:
            print("⚠️ No Stability Data found!")

    def plot_lyapunov_stability(self):
        """
        [Fig 3] Theoretical Stability Proof.
        Shows the monotonic decrease of the Global Lyapunov Function V(t).
        Essential for proving convergence in distributed control.
        """
        if self.stab_df is None: return
        print("📈 Generating Lyapunov Stability Plot...")

        plt.figure(figsize=(8, 5))

        # 绘制总势能 V_total
        sns.lineplot(data=self.stab_df, x='time', y='V_total', label=r'$V_{total}$ (Global Energy)', color='black',
                     linewidth=2)

        # 绘制分项势能 (堆叠区域图效果不佳，改用虚线)
        plt.plot(self.stab_df['time'], self.stab_df['V_track'], '--', label=r'$V_{track}$ (Tracking Error)', alpha=0.7)
        plt.plot(self.stab_df['time'], self.stab_df['V_flow'], ':', label=r'$V_{flow}$ (Flow Entropy)', alpha=0.7)

        # 标注收敛区域
        plt.yscale('log')  # 对数坐标展示收敛率
        plt.title("Theoretical Convergence: Lyapunov Stability Analysis")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Lyapunov Potential (Log Scale)")
        plt.legend()

        plt.savefig("fig_sci_theory_stability.png", dpi=300)
        print("✅ Saved fig_sci_theory_stability.png")

    def plot_cyber_resilience(self):
        """
        [Fig 4] Cyber-Physical Resilience.
        Demonstrates system survival during Communication Loss / Cyber Attack.
        Key Metric: Mode Switching (Performance -> Physics_Fallback).
        """
        if self.exp_df is None: return
        print("🛡️ Generating Cyber-Resilience Plot...")

        # 筛选出一个经历过模式切换的车辆 (通常是 Hauler)
        # 查找 mode 变成 'PHYSICS_FALLBACK' 的时刻
        if 'mode' not in self.exp_df.columns: return

        target_vid = None
        for vid in self.exp_df['id'].unique():
            modes = self.exp_df[self.exp_df['id'] == vid]['mode'].unique()
            if 'PHYSICS_FALLBACK' in modes:
                target_vid = vid
                break

        if not target_vid:
            target_vid = self.exp_df['id'].iloc[0]  # Fallback
            print("Note: No vehicle entered Fallback mode, showing first vehicle.")

        df_v = self.exp_df[self.exp_df['id'] == target_vid].sort_values('time')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Subplot 1: Communication Health (The "Cyber" State)
        ax1.plot(df_v['time'], df_v['comm_health'], color='blue', label='Comm Health')
        ax1.set_ylabel("Link Quality (0-1)")
        ax1.set_title(f"Resilience Analysis for Agent: {target_vid}")
        ax1.grid(True)

        # 标记攻击区间 (Comm < 0.2)
        attack_mask = df_v['comm_health'] < 0.2
        if attack_mask.any():
            ax1.fill_between(df_v['time'], 0, 1, where=attack_mask, color='red', alpha=0.1,
                             label='Cyber Attack / Jamming')
        ax1.legend(loc='upper right')

        # Subplot 2: Velocity & Mode (The "Physical" Response)
        # 用颜色编码模式
        modes = df_v['mode'].unique()
        # 简单的映射：Normal=Green, Fallback=Red, Scout=Orange
        colors = {'STIGMERGY_FOLLOW': 'green', 'ACTIVE_SCOUTING': 'orange', 'PHYSICS_FALLBACK': 'red', 'IDLE': 'grey'}

        # 绘制速度曲线
        ax2.plot(df_v['time'], df_v['vel'], 'k-', alpha=0.3, label='Velocity')

        # 绘制模式散点 (Downsampled)
        sample = df_v.iloc[::10]  # 降采样防止太密
        sns.scatterplot(data=sample, x='time', y='vel', hue='mode', palette=colors, ax=ax2, s=30, edgecolor=None)

        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(title="Control Mode", loc='upper right')

        plt.savefig("fig_sci_resilience.png", dpi=300)
        print("✅ Saved fig_sci_resilience.png")

    def plot_heterogeneity_synergy(self):
        """
        [Fig 5] Heterogeneous Synergy.
        Comparision of Energy Efficiency (SEC) between Scouts and Haulers.
        Shows that Scouts use more energy per kg (to explore), allowing Haulers to save energy.
        """
        if self.exp_df is None: return
        print("🤝 Generating Heterogeneity Plot...")

        # 计算比能耗 SEC = Total Energy / (Mass * Distance)
        summary = self.exp_df.groupby(['id']).agg({
            'energy': 'max',
            'vel': 'mean',
            'time': 'max'
        }).reset_index()

        # 区分类型
        summary['Type'] = summary['id'].apply(lambda x: 'Scout' if 'Scout' in x else 'Hauler')

        # 估算距离 (Vel * Time)
        summary['Distance'] = summary['vel'] * summary['time']
        summary['Mass'] = summary['Type'].apply(lambda x: 1500 if x == 'Scout' else 5000)

        # Specific Energy Consumption (J / kg*m)
        summary['SEC'] = summary['energy'] / (summary['Mass'] * summary['Distance'] + 1.0)

        plt.figure(figsize=(6, 6))
        sns.boxplot(data=summary, x='Type', y='SEC', palette="Set2")
        sns.stripplot(data=summary, x='Type', y='SEC', color='black', alpha=0.5)

        plt.title("Heterogeneous Energy Efficiency")
        plt.ylabel("Specific Energy Consumption (J / kg·m)")
        plt.xlabel("Agent Role")

        plt.savefig("fig_sci_heterogeneity.png", dpi=300)
        print("✅ Saved fig_sci_heterogeneity.png")

    def plot_pareto_final(self):
        """
        [Fig 6] Pareto Frontier (Updated).
        """
        if self.exp_df is None: return
        print("📊 Generating Pareto Plot...")

        summary = self.exp_df.groupby(['id']).agg({
            'time': 'max',
            'energy': 'max',
            'mud_global': 'mean'
        }).reset_index()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=summary, x='time', y='energy',
            hue='mud_global', size='mud_global',
            palette='viridis_r', sizes=(100, 300), alpha=0.9, edgecolor='k'
        )
        plt.title("Pareto Frontier: Efficiency vs Cost")
        plt.xlabel("Mission Duration (s)")
        plt.ylabel("Total Energy (J)")
        plt.legend(title="Mud Factor")
        plt.grid(True, alpha=0.3)

        plt.savefig("fig_sci_pareto_final.png", dpi=300)
        print("✅ Saved fig_sci_pareto_final.png")


if __name__ == "__main__":
    analyzer = DeepSCIAnalyzer()
    analyzer.plot_lyapunov_stability()
    analyzer.plot_cyber_resilience()
    analyzer.plot_heterogeneity_synergy()
    analyzer.plot_pareto_final()
    print("\n🎉 All SCI-Grade figures generated successfully.")