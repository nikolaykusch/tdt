
# train-digital-twin

A discrete-time train digital twin simulator for the comparative analysis of **Edge** and **Cloud** computing architectures in High-Speed Railway (HSR) rolling stock technical condition monitoring systems.

Accompanies the article *"Comparative analysis of Edge and Cloud computing architectures for rolling stock monitoring systems"*. Full compliance of the code with the article's formulas — [`docs/MODEL_DESCRIPTION.md`](docs/MODEL_DESCRIPTION.md).

## What's new in version 2.0

Version 2.0 is a significant expansion of the first version of the model, bridging the gaps between the stated goals of the article and what the code actually calculated:

| # | Added Feature | Reason |
|---|---|---|
| 1 | Economic model (`cost_model.py`): traffic and energy cost | The article promised "Bandwidth Cost" and "energy consumption comparison" — previously, this was entirely missing in the code |
| 2 | Hybrid architecture (`ArchitectureMode.HYBRID_FILTERED`): Edge actually filters data before sending it to the Cloud | Previously, Edge and Cloud received identical raw traffic independently — "traffic savings from Edge" was not supported by anything |
| 3 | Temperature sensors (`TEMP_*` in `config.py`) | Were in the article's abstract, but missing in the code |
| 4 | QoS-prioritization (SAFETY / MONITORING, `queueing.py`) | Previously, all telemetry was processed in a single queue without safety priority |
| 5 | Correct Age of Information for Edge, separately per traffic class | In version 1, `Edge_AoI` was identically zero (bug); in version 1.1, it was discovered that the combined AoI is also misleading — a separate AoI for SAFETY/MONITORING is needed |
| 6 | More realistic radio channel: correlated AR(1) fading, HARQ/BLER, standalone handover duration, triangular SNR drop | Previously, SNR was drawn independently every tick, and the handover lasted exactly one discretization step |
| 7 | Cloud server segment (finite backend throughput + fixed latency) | Previously, all Cloud latency was reduced solely to the radio channel |

Detailed changelog — [`CHANGELOG.md`](CHANGELOG.md).

## Repository Structure

```text
train-digital-twin/
├── src/train_digital_twin/      # main package
│   ├── models.py                # enums and dataclasses (RouteSegment, SimMetrics, ...)
│   ├── config.py                # all model parameters, with documented assumptions
│   ├── queueing.py              # packet queue with QoS priorities (PacketQueue, PriorityQueueSystem)
│   ├── traffic.py               # traffic generation (SAFETY/MONITORING) + Anomaly Burst
│   ├── channel.py               # radio channel: Shannon, Doppler, fading, HARQ, handover
│   ├── edge_node.py             # Edge node: queue, thermal model, energy, ML filtering
│   ├── cloud_node.py            # Cloud: network + server segment, energy
│   ├── cost_model.py            # economics: traffic/energy cost, traffic savings
│   └── simulator.py             # TrainDigitalTwin orchestrator
├── experiments/
│   ├── run_experiment.py        # full factorial design × 2 architecture modes
│   └── make_figures.py          # publication figures based on results
├── tests/                       # pytest, 26 tests
├── docs/
│   └── MODEL_DESCRIPTION.md     # compliance of code and article formulas
├── pyproject.toml
└── requirements.txt

```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .                  # or: pip install -r requirements.txt

```

Requires Python ≥ 3.10.

## Quick Start

```python
from train_digital_twin import TrainDigitalTwin, config
from train_digital_twin.models import ArchitectureMode

route = config.ROUTES['Open terrain (350 km/h)']

twin = TrainDigitalTwin(
    route, run_id=0, scenario_name='demo',
    edge_profile_name='Overload', anomaly_enabled=True,
    architecture_mode=ArchitectureMode.HYBRID_FILTERED,
    seed=42,
)
twin.run()

last = twin.metrics[-1]
print(f"System state: {last.system_state}")
print(f"Edge PLR: {last.edge_plr:.1%}, Cloud PLR: {last.cloud_plr:.1%}")
print(f"Run cost: ${last.cumulative_cost_usd:.2f}")

```

## Reproducing Article Results

```bash
cd experiments
python run_experiment.py --num-runs 5 --out ./results
python make_figures.py --data ./results --out ./figures

```

`run_experiment.py` executes a full factorial design (`config.EXPERIMENT_MATRIX`) in two architecture modes (`PARALLEL_RAW` as a baseline for compatibility with version 1, `HYBRID_FILTERED` as the main realistic mode), saving the results in `raw_results.parquet`. For publication, it is recommended to increase `--num-runs` to 10+ for narrower confidence intervals.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v

```

26 tests cover: the correctness of the packet queue and QoS priorities, economic model, radio channel (tunnel/handover/BLER/LOS vs NLOS), integration scenarios (thermal throttling, tunnel degradation, hybrid traffic savings, correct AoI per class).

## License

MIT — see [`LICENSE`](https://www.google.com/search?q=LICENSE).

```
