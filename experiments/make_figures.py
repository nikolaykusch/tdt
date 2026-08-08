"""
make_figures.py
================
Generates publication figures based on the results of run_experiment.py.

Usage:
    python make_figures.py --data ./results --out ./figures
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10.5,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
})

C_EDGE = "#1f4e79"
C_CLOUD = "#c0504d"
C_SAFETY = "#2e7d32"
C_MONITORING = "#e08e00"

SHORT_NAMES = [
    "1. Open\nterrain, Nominal",
    "2. Deep\ntunnels, Nominal",
    "3. Urban\narea, Nominal",
    "4. Open\nterrain, Overload+Anomaly",
    "5. Mixed\nstress test, Overload+Anomaly",
    "6. Mixed\nstress test, Degraded+Anomaly",
]

ROUTE_SPANS = {
    "Deep tunnels (Connection loss)": [("TUNNEL", 5, 13), ("TUNNEL", 17, 23)],
    "Mixed stress test": [("TUNNEL", 6, 9), ("NLOS", 9, 14), ("TUNNEL", 18, 20)],
    "Urban area (NLOS)": [("NLOS", 0, 16)],
}
SPAN_COLOR = {"TUNNEL": "#888888", "NLOS": "#f0c674"}


def add_spans(ax, scenario):
    for env, a, b in ROUTE_SPANS.get(scenario, []):
        ax.axvspan(a, b, color=SPAN_COLOR[env], alpha=0.15, lw=0)


def base_combo_labels(combo_labels, mode_suffix):
    return [c for c in combo_labels if c.endswith(mode_suffix)]


def make_scenario_overview_figures(df, combo_labels, out_dir):
    hybrid_combos = base_combo_labels(combo_labels, "hybrid_filtered")

    for idx, combo in enumerate(hybrid_combos, start=1):
        sub = df[df['Combination'] == combo]
        scenario = sub['Scenario'].iloc[0]
        grp = sub.groupby('Distance_km').mean(numeric_only=True).reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
        fig.suptitle(f"Scenario {idx} (HYBRID_FILTERED mode): {scenario}", fontsize=11, y=0.99)

        ax = axes[0, 0]
        ax.plot(grp['Distance_km'], grp['Cloud_Latency_s'], color=C_CLOUD, lw=0.9, label='Cloud')
        ax.plot(grp['Distance_km'], grp['Edge_Latency_s'], color=C_EDGE, lw=0.9, label='Edge')
        ax.axhline(0.15, color='black', ls='--', lw=0.7, label='threshold 0.15s')
        ax.set_ylabel("Latency, s"); ax.set_title("Latency", fontsize=10)
        ax.legend(fontsize=7.5); add_spans(ax, scenario)

        ax = axes[0, 1]
        ax.plot(grp['Distance_km'], grp['Cloud_PLR']*100, color=C_CLOUD, lw=1.1, label='Cloud')
        ax.plot(grp['Distance_km'], grp['Edge_PLR']*100, color=C_EDGE, lw=1.1, label='Edge')
        ax.axhline(2.0, color='black', ls='--', lw=0.7, label='threshold 2%')
        ax.set_ylabel("Cumulative PLR, %"); ax.set_title("Packet Loss Ratio", fontsize=10)
        ax.legend(fontsize=7.5); add_spans(ax, scenario)

        ax = axes[1, 0]
        ax.plot(grp['Distance_km'], grp['Edge_Monitoring_AoI_s'], color=C_MONITORING, lw=0.9, label='Edge MONITORING')
        ax.plot(grp['Distance_km'], grp['Edge_Safety_AoI_s'], color=C_SAFETY, lw=0.9, label='Edge SAFETY')
        ax.plot(grp['Distance_km'], grp['Cloud_Monitoring_AoI_s'], color=C_MONITORING, lw=0.9, ls='--', label='Cloud MONITORING')
        ax.set_ylabel("AoI, s"); ax.set_xlabel("Distance, km"); ax.set_title("Age of Information by QoS classes", fontsize=10)
        ax.legend(fontsize=6.5); add_spans(ax, scenario)

        ax = axes[1, 1]
        ax.plot(grp['Distance_km'], grp['Edge_Temperature_C'], color='#8e4b9e', lw=1.0, label='$T_{cpu}$')
        ax.axhline(75, color='black', ls='--', lw=0.7, label='$T_{crit}=75°C$')
        ax.set_ylabel("CPU Temperature, °C"); ax.set_xlabel("Distance, km"); ax.set_title("Edge Thermal Dynamics", fontsize=10)
        ax.legend(fontsize=7.5); add_spans(ax, scenario)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_dir / f"fig_scn{idx}_overview.png")
        plt.close(fig)
        print(f"saved fig_scn{idx}_overview.png")


def make_summary_figures(df, combo_labels, out_dir):
    hybrid_combos = base_combo_labels(combo_labels, "hybrid_filtered")
    raw_combos = base_combo_labels(combo_labels, "parallel_raw")

    # --- Fig 7: state share (hybrid) ---
    STATE_ORDER = ['Both OK', 'Cloud Down', 'Edge Down', 'Both Down']
    COLORS = {'Both OK': '#70ad47', 'Cloud Down': '#9dc3e6', 'Edge Down': '#ffd966', 'Both Down': '#e06666'}
    sub = df[df['Combination'].isin(hybrid_combos)]
    share = sub.groupby(['Combination', 'System_State']).size().unstack(fill_value=0)
    share = share.reindex(columns=STATE_ORDER, fill_value=0)
    share = share.div(share.sum(axis=1), axis=0) * 100
    share = share.reindex(hybrid_combos)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bottom = np.zeros(len(share)); x = np.arange(len(share))
    for state in STATE_ORDER:
        ax.bar(x, share[state].values, bottom=bottom, color=COLORS[state], label=state, width=0.6, edgecolor='white', linewidth=0.5)
        bottom += share[state].values
    ax.set_xticks(x); ax.set_xticklabels(SHORT_NAMES, fontsize=8)
    ax.set_ylabel("Time share, %"); ax.set_ylim(0, 100)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=4, fontsize=9)
    ax.set_title("System State Distribution (HYBRID_FILTERED mode)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "fig7_state_share.png"); plt.close(fig)

    # --- Fig 8: PLR comparison (hybrid) ---
    agg_plr = sub.groupby('Combination').agg(Edge_PLR=('Edge_PLR', 'max'), Cloud_PLR=('Cloud_PLR', 'max')).reindex(hybrid_combos) * 100
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    xw = np.arange(len(agg_plr)); w = 0.35
    ax.bar(xw - w/2, agg_plr['Edge_PLR'], width=w, color=C_EDGE, label='Edge PLR')
    ax.bar(xw + w/2, agg_plr['Cloud_PLR'], width=w, color=C_CLOUD, label='Cloud PLR')
    ax.axhline(2.0, color='black', ls='--', lw=0.8, label='threshold 2%')
    ax.set_xticks(xw); ax.set_xticklabels(SHORT_NAMES, fontsize=8)
    ax.set_ylabel("Final cumulative PLR, %"); ax.legend(fontsize=9)
    ax.set_title("Edge/Cloud PLR Comparison (HYBRID_FILTERED mode)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "fig8_plr_comparison.png"); plt.close(fig)

    # --- Fig 9: cost comparison PARALLEL_RAW vs HYBRID_FILTERED ---
    last_raw = df[df['Combination'].isin(raw_combos)].sort_values('Time_s').groupby(['Combination', 'Simulation_ID']).tail(1)
    last_hyb = df[df['Combination'].isin(hybrid_combos)].sort_values('Time_s').groupby(['Combination', 'Simulation_ID']).tail(1)
    cost_raw = last_raw.groupby('Combination')['Cumulative_Cost_USD'].mean().reindex(raw_combos).values
    cost_hyb = last_hyb.groupby('Combination')['Cumulative_Cost_USD'].mean().reindex(hybrid_combos).values

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    xw = np.arange(len(cost_raw)); w = 0.35
    ax.bar(xw - w/2, cost_raw, width=w, color='#999999', label='PARALLEL_RAW (without filtering)')
    ax.bar(xw + w/2, cost_hyb, width=w, color='#2e7d32', label='HYBRID_FILTERED (with filtering)')
    ax.set_xticks(xw); ax.set_xticklabels(SHORT_NAMES, fontsize=8)
    ax.set_ylabel("Cost per run, USD"); ax.legend(fontsize=9)
    ax.set_title("Traffic and energy cost: without filtering vs hybrid architecture", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "fig9_cost_comparison.png"); plt.close(fig)

    # --- Fig 10: traffic savings % ---
    attempted = last_hyb.groupby('Combination')['Cloud_Bits_Attempted'].mean().reindex(hybrid_combos)
    raw_eq = last_hyb.groupby('Combination')['Cloud_Bits_Raw_Equivalent'].mean().reindex(hybrid_combos)
    savings_pct = (1 - attempted / raw_eq) * 100

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    bars = ax.bar(np.arange(len(savings_pct)), savings_pct.values, color='#2e7d32', width=0.5)
    ax.set_xticks(np.arange(len(savings_pct))); ax.set_xticklabels(SHORT_NAMES, fontsize=8)
    ax.set_ylabel("Traffic savings, %"); ax.set_ylim(0, 100)
    ax.set_title("Traffic savings from Edge filtering (HYBRID_FILTERED vs PARALLEL_RAW)", fontsize=10)
    for b, v in zip(bars, savings_pct.values):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.0f}%", ha='center', fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "fig10_traffic_savings.png"); plt.close(fig)

    print("saved fig7-10 (summary comparisons)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True, help="Directory with raw_results.parquet")
    parser.add_argument("--out", type=str, default="./figures", help="Directory to save figures")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / "raw_results.parquet")
    combo_labels = (data_dir / "combo_labels.txt").read_text().splitlines()

    make_scenario_overview_figures(df, combo_labels, out_dir)
    make_summary_figures(df, combo_labels, out_dir)
    print("All figures saved to", out_dir)


if __name__ == "__main__":
    main()