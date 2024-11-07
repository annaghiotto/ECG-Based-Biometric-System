from abc import ABC
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.fft import dct, fft
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from wfdb import processing
from statsmodels.tsa.ar_model import AutoReg

from FSBase import FSBase
from custom_types import Signal, Features, Template


@dataclass
class FeatureExtractor(ABC, FSBase):
    def __post_init__(self):
        super().__init__()

    def __call__(self, signals: List[Signal]) -> List[Template]:
        return [self.extract(signal) for signal in signals]

    def extract(self, signal: Signal) -> List[Features]:
        pass


class SimpleFeatureExtractor(FeatureExtractor):
    def extract(self, signal: Signal) -> List[Features]:
        return [np.array([
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
        ])]


class StatisticalTimeExtractor(FeatureExtractor):

    def detect_R(self, signal: Signal) -> List[int]:
        r_peaks = processing.gqrs_detect(sig=signal, fs=self.fs)
        return r_peaks

    def extract(self, signal: Signal) -> List[Features]:
        r_peaks = self.detect_R(signal)
        pre_r = int(0.2 * self.fs)
        post_r = int(0.4 * self.fs)

        features = []
        for r_peak in r_peaks:
            start = max(0, r_peak - pre_r)
            end = min(len(signal), r_peak + post_r)
            cycle = signal[start:end]

            target_length = pre_r + post_r
            if len(cycle) < target_length:
                cycle = np.pad(cycle, (0, target_length - len(cycle)), mode='constant')
            else:
                cycle = cycle[:target_length]

            mean = np.mean(cycle)
            std_dev = np.std(cycle)
            p25 = np.percentile(cycle, 25)
            p75 = np.percentile(cycle, 75)
            iqr = p75 - p25
            kurt = kurtosis(cycle)
            skewness = skew(cycle)

            feature_vector = [mean, std_dev, p25, p75, iqr, kurt, skewness]

            features.append(feature_vector)

        try:
            features = np.array(features)
        except ValueError as e:
            print("ValueError during np.array conversion:", e)
            raise

        return features.tolist()


@dataclass
class DiscreteCosineExtractor(FeatureExtractor):
    n_features: int

    def detect_R(self, signal: Signal) -> List[int]:
        r_peaks = processing.gqrs_detect(sig=signal, fs=self.fs)
        return r_peaks

    def autocorrelation(self, signal: Signal, num_coefficients: int) -> List[int]:
        autocorr_result = np.correlate(signal, signal, mode='full')
        autocorr_result = autocorr_result[len(autocorr_result) // 2:]
        return autocorr_result[:num_coefficients]

    def extract(self, signal: Signal) -> List[Features]:
        r_peaks = self.detect_R(signal)
        pre_r = int(0.2 * self.fs)
        post_r = int(0.4 * self.fs)

        features = []  # Initialize as a list
        for r_peak in r_peaks:
            start = max(0, r_peak - pre_r)
            end = min(len(signal), r_peak + post_r)
            cycle = signal[start:end]

            target_length = pre_r + post_r
            if len(cycle) < target_length:
                cycle = np.pad(cycle, (0, target_length - len(cycle)), mode='constant')
            else:
                cycle = cycle[:target_length]

            autocorr_coefficients = self.autocorrelation(cycle, num_coefficients=self.n_features)
            feature_vector = dct(autocorr_coefficients, norm='ortho')
            features.append(feature_vector)  # Use list append

        try:
            features = np.array(features)  # Convert to NumPy array after the loop
            # Optionally, you can verify the shape here
            # print(f"Features shape after conversion to np.array: {features.shape}")
        except ValueError as e:
            print("ValueError during np.array conversion:", e)
            raise

        print(features.shape)

        return features.tolist()


@dataclass
class PCAExtractor(FeatureExtractor):
    """
    Principal Component Analysis (PCA) feature extractor.

    Attributes:
        n_features: int
        Between 0 and min(n_samples, n_features)
    """
    n_features: int

    def detect_R(self, signal: Signal) -> List[int]:
        r_peaks = processing.gqrs_detect(sig=signal, fs=self.fs)
        return r_peaks

    def extract(self, signal: Signal) -> List[Features]:
        r_peaks = self.detect_R(signal)
        pre_r = int(0.2 * self.fs)
        post_r = int(0.4 * self.fs)

        segments = []
        for r_peak in r_peaks:
            start = max(0, r_peak - pre_r)
            end = min(len(signal), r_peak + post_r)
            cycle = signal[start:end]

            target_length = pre_r + post_r
            if len(cycle) < target_length:
                cycle = np.pad(cycle, (0, target_length - len(cycle)), mode='constant')
            else:
                cycle = cycle[:target_length]

            cycle_mean = np.mean(cycle)
            centered_cycle = cycle - cycle_mean

            segments.append(centered_cycle)

        segments_matrix = np.array(segments)

        pca = PCA(n_components=self.n_features)
        principal_components = pca.fit_transform(segments_matrix)

        print(principal_components.shape)

        return principal_components.tolist()


class SARModelExtractor(FeatureExtractor):
    def detect_R(self, signal: Signal) -> List[int]:
        r_peaks = processing.gqrs_detect(sig=signal, fs=self.fs)
        return r_peaks

    def fit_sar_model(self, signal):
        # AR Coefficients calculation
        model = AutoReg(signal, lags=4, old_names=False).fit()
        return model.params[1:]

    def extract(self, signal: Signal) -> List[Features]:
        r_peaks = self.detect_R(signal)
        pre_r = int(0.2 * self.fs)
        post_r = int(0.4 * self.fs)
        
        cycles = []
        for r_peak in r_peaks:
            start = max(0, r_peak - pre_r)
            end = min(len(signal), r_peak + post_r)
            cycle = signal[start:end]
            
            target_length = pre_r + post_r
            if len(cycle) < target_length:
                cycle = np.pad(cycle, (0, target_length - len(cycle)), mode='constant')
            else:
                cycle = cycle[:target_length]
            
            cycles.append(cycle)

        features = []
        n_cycles = 3
        for i in range(0, len(cycles) - 4, n_cycles):
            # Concatenates n_cycles heartbeats in 1 segment
            segment = np.concatenate(cycles[i:i + n_cycles])

            sar_coefficients = self.fit_sar_model(segment)
            features.append(sar_coefficients)

        return features


class StatisticalTimeFreqExtractor(FeatureExtractor):
    
    def detect_R(self, signal: Signal) -> List[int]:
        r_peaks = processing.gqrs_detect(sig=signal, fs=self.fs)
        return r_peaks

    def extract(self, signal: Signal) -> List[Features]:
        r_peaks = self.detect_R(signal)
        pre_r = int(0.2 * self.fs)
        post_r = int(0.4 * self.fs)
        
        features = []
        for r_peak in r_peaks:
            start = max(0, r_peak - pre_r)
            end = min(len(signal), r_peak + post_r)
            cycle = signal[start:end]

            target_length = pre_r + post_r
            if len(cycle) < target_length:
                cycle = np.pad(cycle, (0, target_length - len(cycle)), mode='constant')
            else:
                cycle = cycle[:target_length]

            mean = np.mean(cycle)
            std_dev = np.std(cycle)
            p25 = np.percentile(cycle, 25)
            p75 = np.percentile(cycle, 75)
            iqr = p75 - p25
            kurt = kurtosis(cycle)
            skewness = skew(cycle)

            fft_values = fft(cycle)
            fft_magnitude = np.abs(fft_values[:len(fft_values) // 2])
            fft_power = fft_magnitude ** 2

            mean_power = np.mean(fft_power)
            max_power = np.max(fft_power)
            dominant_frequency = np.argmax(fft_power) * (self.fs / len(cycle))

            feature_vector = [
                mean, std_dev, p25, p75, iqr, kurt, skewness,  # Time domain features
                mean_power, max_power, dominant_frequency       # Frequency domain features
            ]

            features.append(feature_vector)

        try:
            features = np.array(features)
        except ValueError as e:
            print("ValueError during np.array conversion:", e)
            raise

        return features.tolist()