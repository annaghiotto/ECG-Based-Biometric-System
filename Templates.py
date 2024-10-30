from dataclasses import dataclass
import numpy as np
from FeatureExtractor import FeatureExtractor
from Signal import Signal
from Preprocessor import Preprocessor

type Templates = np.ndarray


@dataclass
class TemplatesFactory:
    preprocessor: "Preprocessor"
    feature_extractor: "FeatureExtractor"

    def from_signal(self, signal: Signal) -> Templates:
        return self.feature_extractor(self.preprocessor(signal))
