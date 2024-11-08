from abc import ABC
from dataclasses import dataclass, field
from typing import List, Any

import numpy as np
from numpy import ndarray, dtype
from scipy.fft import dct
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from wfdb import processing
from statsmodels.tsa.ar_model import AutoReg
from FSBase import FSBase
from custom_types import Signal, Features, Template
from hmmlearn import hmm


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
class DCTExtractor(FeatureExtractor):
    n_features: int = 21

    def detect_R(self, signal: Signal) -> List[int]:
        r_peaks = processing.gqrs_detect(sig=signal, fs=self.fs)
        return r_peaks

    def autocorrelation(self, signal: Signal, num_coefficients: int) -> ndarray[Any, dtype[Any]]:
        autocorr_result = np.correlate(signal, signal, mode='same')
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


@dataclass
class HMMExtractor(FeatureExtractor):
    """
    HMM-based feature extractor for ECG signals.
    Extracts features for each ECG segment (heartbeat) using a trained HMM.
    """
    n_components: int = 4  # Number of hidden states
    covariance_type: str = "diag"
    n_iter: int = 100
    model: Any = field(default=None, init=False)
    scaler: Any = field(default=None, init=False)
    fs: int = 250

    def __post_init__(self):
        super().__post_init__()
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            init_params="",
        )
        self.scaler = StandardScaler()

    def detect_R_peaks(self, signal: Signal) -> List[int]:
        """
        Detect R peaks in the ECG signal using the gqrs_detect algorithm.
        """
        r_peaks = processing.gqrs_detect(sig=signal, fs=self.fs)
        return r_peaks

    def segment_heartbeats(self, signal: Signal, r_peaks: List[int],
                           pre_r: int, post_r: int) -> List[np.ndarray]:
        """
        Segment the ECG signal into individual heartbeats based on R peaks.
        Each segment includes 'pre_r' samples before the R peak and 'post_r' samples after.
        """
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

            segments.append(cycle)
        return segments

    def extract(self, signal: Signal) -> List[Features]:
        """
        Extract HMM-based features from the ECG signal.
        Returns a list of feature vectors, one for each heartbeat segment.
        """
        r_peaks = self.detect_R_peaks(signal)
        if len(r_peaks) == 0:
            raise ValueError("No heartbeats detected in the input signal.")

        pre_r = int(0.2 * self.fs)  # 200 ms
        post_r = int(0.4 * self.fs)  # 400 ms

        segments = self.segment_heartbeats(signal, r_peaks, pre_r, post_r)

        flattened_segments = [segment.reshape(-1, 1) for segment in segments]
        concatenated = np.concatenate(flattened_segments)

        # Normalize the data
        self.scaler.fit(concatenated)
        scaled_data = self.scaler.transform(concatenated)

        self.model.fit(scaled_data)

        features_list = []
        for segment in segments:
            segment_scaled = self.scaler.transform(segment.reshape(-1, 1))

            log_likelihood = self.model.score(segment_scaled)
            posteriors = self.model.predict_proba(segment_scaled)
            mean_posteriors = np.mean(posteriors, axis=0).tolist()
            state_sequence = self.model.predict(segment_scaled)
            state_counts = np.bincount(state_sequence, minlength=self.model.n_components).tolist()

            feature_vector = np.array([log_likelihood, *mean_posteriors, *state_counts])
            features_list.append(feature_vector)

        return features_list
