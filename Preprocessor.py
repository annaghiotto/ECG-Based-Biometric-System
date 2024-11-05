from abc import ABC
from typing import List

from FSBase import FSBase
from Signal import Signal
from scipy.signal import butter, lfilter, iirnotch


class Preprocessor(ABC, FSBase):
    def __call__(self, signals: List[Signal]) -> List[Signal]:
        return [self.preprocess(signal) for signal in signals]

    def preprocess(self, signal: Signal) -> Signal:
        pass


class SimplePreprocessor(Preprocessor):
    def preprocess(self, signal: Signal) -> Signal:
        return signal


class BasicPreprocessor(Preprocessor):

    def preprocess(self, signal: Signal) -> Signal:
        signal = self.bandpass_filter(signal, 0.5, 30.0)
        signal = self.notch_filter(signal, 50)
        return signal

    def bandpass_filter(self, signal: Signal, lower_fr: float, higher_fr: float) -> Signal:
        w_low = lower_fr * 2 / self.fs
        w_high = higher_fr * 2 / self.fs
        b, a = butter(N=4, Wn=[w_low, w_high], btype='band')
        return lfilter(b, a, signal)

    def notch_filter(self, signal: Signal, freq: float, quality_factor: float = 30.0) -> Signal:
        w0 = freq * 2 / self.fs
        b, a = iirnotch(w0, quality_factor)
        return lfilter(b, a, signal)