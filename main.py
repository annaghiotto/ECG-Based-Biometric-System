from experiment import run_experiments
from Preprocessor import BasicPreprocessor, SARModelPreprocessor
from FeatureExtractor import StatisticalTimeExtractor, DCTExtractor, PCAExtractor, SARModelExtractor, StatisticalTimeFreqExtractor, HMMExtractor

if __name__ == "__main__":
    run_experiments(
        preprocessor_feature_combinations=[
            (BasicPreprocessor(), StatisticalTimeExtractor()),
            (BasicPreprocessor(), DCTExtractor()),
            (BasicPreprocessor(), PCAExtractor(5)),
            (SARModelPreprocessor(), SARModelExtractor()),
            (BasicPreprocessor(), StatisticalTimeFreqExtractor()),
            (BasicPreprocessor(), HMMExtractor())
        ],
        n_folds=3
    )