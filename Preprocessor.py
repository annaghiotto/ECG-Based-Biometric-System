from abc import ABC
from typing import List

import numpy as np

from FSBase import FSBase
from custom_types import Signal
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
    

class SARModelPreprocessor(Preprocessor):
    def preprocess(self, signal: Signal) -> Signal:
        signal = self.highpass_filter(signal, 2.)
        signal = self.notch_filter(signal, 50)
        return signal
    
    def highpass_filter(self, signal: Signal, cutoff: float) -> Signal:
        w_cut = cutoff * 2 / self.fs
        b, a = butter(N=1, Wn=w_cut, btype='high', analog=False)
        return lfilter(b, a, signal)
    
    def notch_filter(self, signal: Signal, freq: float, quality_factor: float = 30.0) -> Signal:
        w0 = freq * 2 / self.fs
        b, a = iirnotch(w0, quality_factor)
        return lfilter(b, a, signal)


class PanTompkinsPreprocessor(Preprocessor):

    def preprocess(self, signal: Signal) -> Signal:
        signal = self.bandpass_filter(signal, 5., 15., )
        signal = self.derivative_filter(signal)
        signal = self.squaring(signal)
        signal = self.moving_window_integration(signal, window_size=int(0.15 * self.fs))
        return signal
    
    def bandpass_filter(self, signal: Signal, lower_fr: float, higher_fr: float) -> Signal:
        w_low = lower_fr * 2 / self.fs
        w_high = higher_fr * 2 / self.fs
        b, a = butter(N=2, Wn=[w_low, w_high], btype='band')
        return lfilter(b, a, signal)
    
    def derivative_filter(self, signal: Signal) -> Signal:
        return np.diff(signal, prepend=signal[0])
    
    def squaring(self, signal: Signal) -> Signal:
        return np.power(signal, 2)
    
    def moving_window_integration(self, signal: Signal, window_size: int = None) -> Signal:
        if window_size is None:
            window_size = int(0.08 * self.fs)

        cumulative_sum = np.cumsum(signal)
        integrated_signal = np.zeros_like(signal)
        integrated_signal[window_size:] = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        integrated_signal[:window_size] = cumulative_sum[:window_size] / np.arange(1, window_size + 1)
        
        return integrated_signal 

"""class PanTompkinsPreprocessor(Preprocessor):

    def preprocess(self, signal: Signal) -> Signal:
        signal = self.bandpass_filter(signal, 5., 15.)
        signal = self.derivative_filter(signal)
        signal = self.squaring(signal)
        signal = self.moving_window_integration(signal)
        return signal
    
    def bandpass_filter(self, signal: Signal, lower_fr: float, higher_fr: float) -> Signal:
        w_low = lower_fr * 2 / self.fs
        w_high = higher_fr * 2 / self.fs
        b, a = butter(N=2, Wn=[w_low, w_high], btype='band')
        return lfilter(b, a, signal)

    def derivative_filter(self, signal: Signal) -> Signal:
        return np.diff(signal, prepend=signal[0])

    def squaring(self, signal: Signal) -> Signal:
        return np.power(signal, 2)

    def moving_window_integration(self, signal: Signal, window_size: int = None) -> Signal:
        if window_size is None:
            window_size = int(0.08 * self.fs)

        cumulative_sum = np.cumsum(signal)
        integrated_signal = np.zeros_like(signal)
        integrated_signal[window_size:] = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        integrated_signal[:window_size] = cumulative_sum[:window_size] / np.arange(1, window_size + 1)
        
        return integrated_signal"""