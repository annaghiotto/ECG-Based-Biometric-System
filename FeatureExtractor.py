from abc import ABC
from typing import List

import numpy as np

from Signal import Signal

type Features = np.ndarray


class FeatureExtractor(ABC):
    def __call__(self, signals: List[Signal]) -> List[Features]:
        return [self.extract(signal) for signal in signals]

    def extract(self, signal: Signal) -> Signal:
        pass


class SimpleFeatureExtractor(FeatureExtractor):
    def extract(self, signal: Signal) -> Features:
        return np.array([
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
        ])
