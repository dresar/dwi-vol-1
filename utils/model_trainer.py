from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class TrainArtifacts:
    model_path: str
    shap_summary_path: str
    metrics: dict
    confusion: dict
    feature_importance: dict
    classes: list[str]
    feature_names: list[str]
    dataset_id: str
    trained_at: str
    duration_seconds: float


ProgressCb = Callable[[int, str], None]


def noop_progress(_: int, __: str) -> None:
    return

