from experiment import run_experiments
from Preprocessor import BasicPreprocessor, SARModelPreprocessor
from FeatureExtractor import StatisticalTimeExtractor, DCTExtractor, PCAExtractor, SARModelExtractor, StatisticalTimeFreqExtractor, HMMExtractor

if __name__ == "__main__":
    # Run experiments with different combinations of preprocessors and feature extractors
    run_experiments(
        preprocessor_feature_combinations=[
            (BasicPreprocessor(), StatisticalTimeExtractor()),    # Time-domain statistical feature extraction
            (BasicPreprocessor(), DCTExtractor()),                # DCT-based feature extraction
            (BasicPreprocessor(), PCAExtractor(5)),               # PCA-based feature extraction with 5 components
            (SARModelPreprocessor(), SARModelExtractor()),        # SAR model-based feature extraction
            (BasicPreprocessor(), StatisticalTimeFreqExtractor()),# Time-frequency statistical feature extraction
            (BasicPreprocessor(), HMMExtractor())                 # HMM-based feature extraction
        ],
        n_folds=3  # Use 3-fold cross-validation
    )
