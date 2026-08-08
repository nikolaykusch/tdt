"""
run_experiment.py
==================
Executes the full factorial design (config.EXPERIMENT_MATRIX) in TWO architecture
modes (ArchitectureMode.PARALLEL_RAW and HYBRID_FILTERED) for each
factor combination, with config.NUM_RUNS Monte-Carlo replications each.

Result — a raw_results.parquet file (full array per tick) and
combo_labels.txt (scenario order) in the current working directory,
which are then used by make_figures.py.

Usage:
    python run_experiment.py [--num-runs N] [--out DIR]
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from train_digital_twin import config
from train_digital_twin.simulator import TrainDigitalTwin
from train_digital_twin.models import ArchitectureMode

# Mapping of English SimMetrics fields (models.py) to column names
# in the reporting dataset. Renaming is done ONCE, on the already
# built DataFrame, rather than row by row — this is orders of magnitude more
# memory efficient than building a list of dictionaries with renamed keys at each tick.
COLUMN_RENAME = {
    'scenario': 'Scenario',
    'edge_profile': 'Edge_Profile',
    'architecture_mode': 'Architecture',
    'anomaly_on': 'Anomaly_Mode',
    'run_id': 'Simulation_ID',
    'timestamp': 'Time_s',
    'speed': 'Speed_km_h',
    'env_type': 'Environment_Type',

    'edge_temp': 'Edge_Temperature_C',
    'edge_latency': 'Edge_Latency_s',
    'edge_plr': 'Edge_PLR',
    'edge_aoi': 'Edge_AoI_s',
    'edge_safety_aoi': 'Edge_Safety_AoI_s',
    'edge_monitoring_aoi': 'Edge_Monitoring_AoI_s',
    'edge_safety_plr': 'Edge_Safety_PLR',
    'edge_monitoring_plr': 'Edge_Monitoring_PLR',
    'edge_energy_j': 'Edge_Energy_J',

    'cloud_harq_bler': 'Cloud_BLER',
    'cloud_latency': 'Cloud_Latency_s',
    'cloud_server_latency': 'Cloud_Server_Latency_s',
    'cloud_plr': 'Cloud_PLR',
    'cloud_aoi': 'Cloud_AoI_s',
    'cloud_safety_aoi': 'Cloud_Safety_AoI_s',
    'cloud_monitoring_aoi': 'Cloud_Monitoring_AoI_s',
    'cloud_safety_plr': 'Cloud_Safety_PLR',
    'cloud_monitoring_plr': 'Cloud_Monitoring_PLR',

    'cloud_bits_sent': 'Cloud_Bits_Sent',
    'cloud_bits_attempted': 'Cloud_Bits_Attempted',
    'cloud_bits_raw_equivalent': 'Cloud_Bits_Raw_Equivalent',
    'edge_energy_cumulative_j': 'Edge_Energy_Cumulative_J',
    'cloud_tx_energy_cumulative_j': 'Cloud_TX_Energy_Cumulative_J',
    'cumulative_cost_usd': 'Cumulative_Cost_USD',

    'anomaly_active': 'Anomaly_Active',
    'system_state': 'System_State',
}


def run_experiment(num_runs: int, out_dir: Path) -> None:
    per_run_frames = []
    combo_labels = []
    t0 = time.time()

    architecture_modes = [ArchitectureMode.PARALLEL_RAW, ArchitectureMode.HYBRID_FILTERED]

    for combo in config.EXPERIMENT_MATRIX:
        route_name = combo['route']
        edge_profile = combo['edge_profile']
        anomaly = combo['anomaly']
        route_data = config.ROUTES[route_name]

        for arch_mode in architecture_modes:
            combo_label = (
                f"{route_name} | {edge_profile}"
                f"{' | Anomaly' if anomaly else ''} | {arch_mode.value}"
            )
            combo_labels.append(combo_label)

            for run_id in range(num_runs):
                seed = hash((route_name, edge_profile, anomaly, arch_mode.value, run_id)) & 0xFFFFFFFF
                twin = TrainDigitalTwin(
                    route_data, run_id,
                    scenario_name=route_name,
                    edge_profile_name=edge_profile,
                    anomaly_enabled=anomaly,
                    architecture_mode=arch_mode,
                    seed=seed,
                )
                twin.run()

                run_df = pd.DataFrame([m.__dict__ for m in twin.metrics])
                run_df['Combination'] = combo_label
                run_df['Distance_km'] = run_df['distance'] / 1000.0
                run_df.drop(columns=['distance'], inplace=True)
                per_run_frames.append(run_df)

                print(f"done: {combo_label} run {run_id + 1}/{num_runs}, rows: {len(run_df)}")

    print(f"Simulation time: {time.time() - t0:.1f}s — concatenating {len(per_run_frames)} run-frames...")

    df = pd.concat(per_run_frames, ignore_index=True)
    per_run_frames.clear()  # freeing memory as soon as possible

    df.rename(columns=COLUMN_RENAME, inplace=True)
    df.rename(columns={
        'edge_queue_bits': 'Edge_Queue_Bits',
        'cloud_queue_bits': 'Cloud_Queue_Bits',
        'cloud_capacity': 'Cloud_Capacity_bps',
    }, inplace=True)
    df['Cloud_Capacity_Mbps'] = df['Cloud_Capacity_bps'] / 1e6

    print(f"Total time: {time.time() - t0:.1f}s, total rows: {len(df)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "raw_results.parquet")
    (out_dir / "combo_labels.txt").write_text("\n".join(dict.fromkeys(combo_labels)))
    print(df.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-runs", type=int, default=config.NUM_RUNS,
                        help="Number of Monte-Carlo replications per combination (default from config.py)")
    parser.add_argument("--out", type=str, default=".", help="Directory to save results")
    args = parser.parse_args()

    run_experiment(args.num_runs, Path(args.out))