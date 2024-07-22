from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: float = None
    mse: float = None
    left: Node = None
    right: Node = None