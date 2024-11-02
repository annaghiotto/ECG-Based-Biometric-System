from abc import ABC
from typing import List
from Signal import Signal


class Preprocessor(ABC):
    def __call__(self, signals: List[Signal]) -> List[Signal]:
        return [self.preprocess(signal) for signal in signals]

    def preprocess(self, signal: Signal) -> Signal:
        pass


class SimplePreprocessor(Preprocessor):
    def preprocess(self, signal: Signal) -> Signal:
        return signal
