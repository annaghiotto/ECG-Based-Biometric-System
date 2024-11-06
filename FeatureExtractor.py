from abc import ABC
from typing import List

import numpy as np
from scipy.fft import dct
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from wfdb import processing

from FSBase import FSBase
from custom_types import Signal, Features, Template


class FeatureExtractor(ABC, FSBase):
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


class DiscreteCosineExtractor(FeatureExtractor):

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

            autocorr_coefficients = self.autocorrelation(cycle, num_coefficients=21)
            feature_vector = dct(autocorr_coefficients, norm='ortho')
            features.append(feature_vector)  # Use list append

        try:
            features = np.array(features)  # Convert to NumPy array after the loop
            # Optionally, you can verify the shape here
            # print(f"Features shape after conversion to np.array: {features.shape}")
        except ValueError as e:
            print("ValueError during np.array conversion:", e)
            raise

        return features.tolist()
    

class PCAExtractor(FeatureExtractor):

    def detect_R(self, signal: Signal) -> List[int]:
        r_peaks = processing.gqrs_detect(sig=signal, fs=self.fs)
        return r_peaks
    
    def extract(self, signal: Signal) -> List[Features]:
        # Rileva i picchi R
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

        pca = PCA(n_components=min(segments_matrix.shape[0], segments_matrix.shape[1]))
        principal_components = pca.fit_transform(segments_matrix)

        return principal_components.tolist()

    