"""
train_digital_twin
===================
Discrete-time train digital twin simulator for comparative analysis of
Edge and Cloud computing architectures in HSR rolling stock monitoring
systems.

Public API:
    >>> from train_digital_twin import TrainDigitalTwin, config
    >>> from train_digital_twin.models import ArchitectureMode
    >>> route = config.ROUTES['Open terrain (350 km/h)']
    >>> twin = TrainDigitalTwin(route, run_id=0, scenario_name='demo',
    ...                         edge_profile_name='Nominal', anomaly_enabled=False,
    ...                         architecture_mode=ArchitectureMode.HYBRID_FILTERED,
    ...                         seed=42)
    >>> twin.run()
    >>> len(twin.metrics) > 0
    True

Detailed description of all formulas and their correspondence to the article's equations —
docs/MODEL_DESCRIPTION.md.
"""

from .simulator import TrainDigitalTwin
from . import config
from . import models
from . import cost_model

__all__ = ["TrainDigitalTwin", "config", "models", "cost_model"]

__version__ = "2.0.0"