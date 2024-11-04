from abc import ABC
from typing import List

import numpy as np
from Signal import Signal
from scipy.signal import butter, lfilter, iirnotch


class Preprocessor(ABC):
    def __call__(self, signals: List[Signal], fs: float = None) -> List[Signal]:
        return [self.preprocess(signal, fs) for signal in signals]

    def preprocess(self, signal: Signal, fs: float) -> Signal:
        pass


class SimplePreprocessor(Preprocessor):
    def preprocess(self, signal: Signal, fs: float = None) -> Signal:
        return signal


class BasicPreprocessor(Preprocessor):
    
    def  preprocess(self, signal: Signal, fs: float) -> Signal:
        signal = self.bandpass_filter(signal, fs, 0.5, 40)
        signal = self.notch_filter(signal, fs, 50)
        signal = self.normalize(signal)
        return signal
    
    def bandpass_filter(self, signal: Signal, fs: float, lower_fr: float, higher_fr: float)  -> Signal:
        w_low = lower_fr*2/fs
        w_high = higher_fr*2/fs
        b, a = butter(N=4, Wn=[w_low, w_high], btype='band')
        return  lfilter(b, a, signal)
    
    def notch_filter(self, signal: Signal, fs: float, freq: float, quality_factor: float = 30.0) -> Signal:
        w0 = freq*2/fs
        b, a = iirnotch(w0, quality_factor)
        return lfilter(b, a, signal)
    
    def normalize(self, signal: Signal) -> Signal:
        min_val = np.min(signal)
        max_val = np.max(signal)
        return (signal - min_val)/(max_val - min_val)