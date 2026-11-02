"""The result format every check returns: one row per check per episode."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import pandas as pd


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    dataset: str
    episode_index: int
    check_name: str
    status: Status
    metric: float
    threshold: float
    details: str = ""


def results_to_frame(results: list[CheckResult]) -> pd.DataFrame:
    rows = [{**asdict(r), "status": r.status.value} for r in results]
    return pd.DataFrame(rows)