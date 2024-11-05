from abc import ABC
from itertools import chain
from typing import List

import numpy as np
from scipy.stats import kurtosis, skew
from wfdb import processing

from FSBase import FSBase
from Signal import Signal

type Features = np.ndarray


class FeatureExtractor(ABC, FSBase):
    def __call__(self, signals: List[Signal]) -> List[Features]:
        return list(chain.from_iterable(self.extract(signal) for signal in signals))

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
            # print(f"Feature vector length: {len(feature_vector)} - Values: {feature_vector}")

            features.append(feature_vector)
        #
        # lengths = [len(f) for f in features]
        # print(f"Feature lengths: {lengths}")

        try:
            features = np.array(features)
            # print(f"Features shape after conversion to np.array: {features.shape}")
        except ValueError as e:
            print("ValueError during np.array conversion:", e)
            raise

        return features.tolist()
