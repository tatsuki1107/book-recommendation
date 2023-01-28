from dataclasses import dataclass
import pandas as pd
from typing import Dict, List


@dataclass(frozen=True)
class RecommendResult:
    rating: pd.DataFrame
    #user2items: Dict[int, List[int]]


@dataclass(frozen=True)
class Metrics:
    rmse: float
    #precision_at_k: float
    #recall_at_k: float

    def __repr__(self):
        return f"rmse={self.rmse:.3f}"
