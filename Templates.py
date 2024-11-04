from dataclasses import dataclass
from typing import List

import numpy as np
from FeatureExtractor import FeatureExtractor
from Signal import Signal
from Preprocessor import Preprocessor

type Templates = np.ndarray


@dataclass
class TemplatesFactory:
    preprocessor: "Preprocessor"
    feature_extractor: "FeatureExtractor"

    def from_signals(self, signals: List[Signal], fs: float) -> Templates:
        preprocessed_signals = self.preprocessor(signals, fs)
        features = [self.feature_extractor.extract(signal) for signal in preprocessed_signals]
        return np.array(features).reshape(len(features), -1)
