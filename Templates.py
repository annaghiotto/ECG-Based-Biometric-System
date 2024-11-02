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

    def from_signals(self, signals: List[Signal]) -> Templates:
        return self.feature_extractor(self.preprocessor(signals))
