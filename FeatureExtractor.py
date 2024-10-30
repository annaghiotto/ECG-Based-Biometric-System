from abc import ABC
import numpy as np
from Signal import Signal


class FeatureExtractor(ABC):
    def __call__(self, signals: np.ndarray) -> Signal:
        pass
