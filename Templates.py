from dataclasses import dataclass
from typing import List

from FeatureExtractor import FeatureExtractor, Features
from Signal import Signal
from Preprocessor import Preprocessor

type Templates = List[Features]


@dataclass
class TemplatesFactory:
    preprocessor: "Preprocessor"
    feature_extractor: "FeatureExtractor"

    def from_signals(self, signals: List[Signal]) -> Templates:
        return self.feature_extractor(self.preprocessor(signals))
