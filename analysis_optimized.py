import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

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
    Generates 5 Key Figures for "Physics-Aware Distributed Edge Control":
    1. Theoretical Proof: Lyapunov Stability Convergence (Fig 3).
    2. Resilience: System behavior under Cyber-Attack (Fig 4).
    3. Heterogeneity: Synergy between Scouts and Haulers (Fig 5).
    4. Benchmark: Optimality Gap vs God-Mode Oracle (Fig 6).
    5. Performance: Pareto Efficiency Frontier (Fig 7).
    """

    def __init__(self):
        self.single_run_df = None
        self.stab_df = None
        self.batch_df = None
        self.oracle_df = None
        self._load_data()

    def _load_data(self):
        print(f"📂 Loading data from {DATA_DIR}...")

        # 1. Load Single Run Data (from main.py)
        exp_files = glob.glob(os.path.join(DATA_DIR, "SCI_Exp_*.csv"))
        if exp_files:
            latest = max(exp_files, key=os.path.getmtime)
            self.single_run_df = pd.read_csv(latest)
            print(f"  - Single Run: {os.path.basename(latest)}")

        stab_files = glob.glob(os.path.join(DATA_DIR, "SCI_Stability_*.csv"))
        if stab_files:
            latest = max(stab_files, key=os.path.getmtime)
            self.stab_df = pd.read_csv(latest)
            print(f"  - Stability:  {os.path.basename(latest)}")

        # 2. Load Batch Data (from experiment_runner.py)
        batch_file = os.path.join(DATA_DIR, "batch_results_sci.csv")
        if os.path.exists(batch_file):
            self.batch_df = pd.read_csv(batch_file)
            print(f"  - Batch Data: batch_results_sci.csv")

        # 3. Load Oracle Data
        oracle_file = os.path.join(DATA_DIR, "oracle_baseline.csv")
        if os.path.exists(oracle_file):
            self.oracle_df = pd.read_csv(oracle_file)
            print(f"  - Oracle Data: oracle_baseline.csv")

    def plot_lyapunov_stability(self):
        """[Fig 3] Theoretical Stability Proof."""
        if self.stab_df is None: return
        print("📈 Painting Lyapunov Stability...")

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=self.stab_df, x='time', y='V_total', label=r'$V_{total}$ (Global Potential)', color='black',
                     lw=2)
        plt.plot(self.stab_df['time'], self.stab_df['V_track'], '--', label=r'$V_{track}$ (Tracking Error)', alpha=0.7)
        plt.plot(self.stab_df['time'], self.stab_df['V_flow'], ':', label=r'$V_{flow}$ (Flow Entropy)', alpha=0.7)

        plt.yscale('log')
        plt.title("Theoretical Convergence: Lyapunov Stability Analysis")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Lyapunov Function V(t) [Log Scale]")
        plt.legend()
        plt.savefig("fig_sci_theory_stability.png", dpi=300)

    def plot_cyber_resilience(self):
        """[Fig 4] Resilience under Attack."""
        if self.single_run_df is None: return
        print("🛡️ Painting Cyber-Resilience...")

        # Find a vehicle that entered fallback mode
        target_vid = None
        for vid in self.single_run_df['id'].unique():
            if 'PHYSICS_FALLBACK' in self.single_run_df[self.single_run_df['id'] == vid]['mode'].values:
                target_vid = vid
                break

        if not target_vid: target_vid = self.single_run_df['id'].iloc[0]
        df = self.single_run_df[self.single_run_df['id'] == target_vid].sort_values('time')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot Comm Health
        ax1.plot(df['time'], df['comm_health'], 'b-', label='Comm Health')
        ax1.set_ylabel("Link Quality")
        ax1.set_title(f"Resilience Analysis (Agent: {target_vid})")
        ax1.grid(True)
        # Highlight Attack
        attack = df['comm_health'] < 0.2
        if attack.any():
            ax1.fill_between(df['time'], 0, 1, where=attack, color='red', alpha=0.1, label='Jamming/Attack')
        ax1.legend(loc='upper right')

        # Plot Velocity & Mode
        ax2.plot(df['time'], df['vel'], 'k-', alpha=0.3, label='Velocity')
        colors = {'STIGMERGY_FOLLOW': 'green', 'ACTIVE_SCOUTING': 'orange', 'PHYSICS_FALLBACK': 'red', 'IDLE': 'grey'}
        sns.scatterplot(data=df.iloc[::5], x='time', y='vel', hue='mode', palette=colors, ax=ax2, s=40, edgecolor=None)

        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(title="Control Mode", loc='upper right')
        plt.savefig("fig_sci_resilience.png", dpi=300)

    def plot_heterogeneity_synergy(self):
        """[Fig 5] Heterogeneous Efficiency."""
        if self.single_run_df is None: return
        print("🤝 Painting Heterogeneity...")

        # Calculate SEC for each agent
        agg = self.single_run_df.groupby('id').agg({'energy': 'max', 'vel': 'mean', 'time': 'max'}).reset_index()
        agg['Type'] = agg['id'].apply(lambda x: 'Scout' if 'Scout' in x else 'Hauler')
        agg['Mass'] = agg['Type'].apply(lambda x: 1500 if x == 'Scout' else 5000)
        agg['Dist'] = agg['vel'] * agg['time']
        agg['SEC'] = agg['energy'] / (agg['Mass'] * agg['Dist'] + 1.0)

        plt.figure(figsize=(6, 5))
        sns.boxplot(data=agg, x='Type', y='SEC', palette="Set2")
        sns.stripplot(data=agg, x='Type', y='SEC', color='black', alpha=0.5)
        plt.title("Energy Efficiency by Role")
        plt.ylabel("Specific Energy Consumption (J / kg·m)")
        plt.savefig("fig_sci_heterogeneity.png", dpi=300)

    def plot_optimality_gap(self):
        """[Fig 6] Optimality Gap (PADR vs Oracle)."""
        if self.batch_df is None or self.oracle_df is None: return
        print("📊 Painting Optimality Gap...")

        # Filter PADR results (Heavy Hauler)
        hauler_df = self.batch_df[self.batch_df['id'].str.contains("Heavy_Hauler")]
        padr_stats = hauler_df.groupby('mud').agg({'energy': 'mean'}).reset_index()

        plt.figure(figsize=(8, 6))
        sns.lineplot(data=padr_stats, x='mud', y='energy', marker='o', label='PADR (Distributed)', color='blue', lw=2)
        sns.lineplot(data=self.oracle_df, x='mud', y='oracle_energy', marker='X', label='Oracle (Lower Bound)',
                     color='red', linestyle='--', lw=2)

        plt.fill_between(padr_stats['mud'], padr_stats['energy'], self.oracle_df['oracle_energy'], color='gray',
                         alpha=0.1, label='Optimality Gap')

        plt.title("Optimality Gap Analysis")
        plt.xlabel("Mud Factor (Environmental Complexity)")
        plt.ylabel("Total Energy Consumption (J)")
        plt.legend()
        plt.savefig("fig_sci_optimality_gap.png", dpi=300)

    def plot_pareto_final(self):
        """[Fig 7] Pareto Frontier."""
        if self.batch_df is None: return
        print("📉 Painting Pareto Frontier...")

        summary = self.batch_df.groupby(['id', 'exp_id']).agg({
            'time': 'max', 'energy': 'max', 'mud': 'mean'
        }).reset_index()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=summary, x='time', y='energy',
            hue='mud', size='mud',
            palette='viridis_r', sizes=(50, 200), alpha=0.8, edgecolor='k'
        )
        plt.title("Pareto Efficiency Frontier")
        plt.xlabel("Mission Time (s)")
        plt.ylabel("Total Energy (J)")
        plt.legend(title="Mud Factor", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig("fig_sci_pareto_final.png", dpi=300)


if __name__ == "__main__":
    analyzer = DeepSCIAnalyzer()
    analyzer.plot_lyapunov_stability()
    analyzer.plot_cyber_resilience()
    analyzer.plot_heterogeneity_synergy()
    analyzer.plot_optimality_gap()
    analyzer.plot_pareto_final()
    print("\n🎉 All figures generated.")