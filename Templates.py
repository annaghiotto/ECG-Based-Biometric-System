from dataclasses import dataclass
from typing import List

from FeatureExtractor import FeatureExtractor
from custom_types import Signal, Template
from Preprocessor import Preprocessor


@dataclass
class TemplatesFactory:
    preprocessor: "Preprocessor"
    feature_extractor: "FeatureExtractor"

    def from_signals(self, signals: List[Signal]) -> List[Template]:
        return self.feature_extractor(self.preprocessor(signals))
