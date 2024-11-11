from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars
from Classifier import XGBoostClassifier
from DataSource import GetSBData, GetEcgIDData
from utils import train_test_split, k_fold_split
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count


def plot_results(results_df: pd.DataFrame, filename="experiment_results.png"):
    """
    Plot the experiment results as bar charts for multiple metrics on a single plot.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing experiment results.
        filename (str): The file path where the plot image will be saved.
    """
    sns.set(style="whitegrid")

    # Define metrics and titles for display
    metrics = ['accuracy', 'eer', 'auc']
    metric_titles = {
        'accuracy': 'Accuracy',
        'eer': 'Equal Error Rate (EER)',
        'auc': 'Area Under ROC Curve (AUC)'
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=False)

    # Generate bar plots for each metric
    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=results_df,
            x='feature_extractor',
            y=metric,
            hue='data_source',  # Use data source as hue
            ax=ax
        )
        ax.set_title(f'Comparison of {metric_titles[metric]}', fontsize=16)
        ax.set_xlabel('Feature Extractor', fontsize=14)
        ax.set_ylabel(metric_titles[metric], fontsize=14)
        ax.legend(title='Data Source', fontsize=10, title_fontsize=12)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# Usage example:
# plot_results(results_df, filename="experiment_results.png")


@dataclass
class ExperimentResult:
    data_source: str
    preprocessor: str
    feature_extractor: str
    accuracy: Optional[float]
    eer: Optional[float]
    auc: Optional[float]


def process_combination(args):
    """
    Processes a single combination of preprocessor and feature extractor.
    Returns the ExperimentResult for each data source.
    """
    preprocessor, feature_extractor, n_folds = args
    preprocessor_name = type(preprocessor).__name__
    feature_extractor_name = type(feature_extractor).__name__

    results = []

    # Load data for both ECGID and SB sources
    data_ecg_id = list(GetEcgIDData('ecg-id-database-1.0.0', preprocessor, feature_extractor).generator())
    data_sb = list(GetSBData('SB_ECGDatabase_01', preprocessor, feature_extractor).generator())

    for data_source_name, data in [("ECGID", data_ecg_id), ("SB", data_sb)]:
        # Handle k-fold cross-validation
        if n_folds == 1:
            try:
                train_data, test_data = train_test_split(data, test_size=0.2)
                folds = [(train_data, test_data)]
            except Exception as e:
                print(f"Error during train-test split for {data_source_name}: {e}")
                continue
        else:
            try:
                folds = k_fold_split(data, n_folds)
            except Exception as e:
                print(f"Error during k-fold split for {data_source_name}: {e}")
                continue

        accuracy_list = []
        eer_list = []
        auc_list = []

        # Evaluate model on each fold
        for fold_num, (train, test) in enumerate(folds, start=1):
            try:
                cloned_classifier = XGBoostClassifier(threshold=0.5)
                cloned_classifier.fit(train, test)
                accuracy, eer, eer_threshold, auc_score = cloned_classifier.evaluate(test)
                accuracy_list.append(accuracy)
                eer_list.append(eer)
                auc_list.append(auc_score)
            except Exception as e:
                print(f"Error during evaluation in fold {fold_num} for {data_source_name}: {e}")
                continue

        # Calculate average scores across folds
        avg_accuracy = np.mean(accuracy_list) if accuracy_list else None
        avg_eer = np.mean(eer_list) if eer_list else None
        avg_auc = np.mean(auc_list) if auc_list else None

        result = ExperimentResult(
            data_source=data_source_name,
            preprocessor=preprocessor_name,
            feature_extractor=feature_extractor_name,
            accuracy=avg_accuracy,
            eer=avg_eer,
            auc=avg_auc
        )
        results.append(result)

    return results


def run_experiments(preprocessor_feature_combinations: List[Tuple], n_folds: int = 1) -> pd.DataFrame:
    """
    Run experiments for each combination of Preprocessor and FeatureExtractor across
    SB and ECG-ID data sources using multiprocessing.

    Parameters:
        preprocessor_feature_combinations (List[Tuple]): List of (Preprocessor, FeatureExtractor) tuples.
        n_folds (int): Number of folds for cross-validation. Default is 1 (train-test split).

    Returns:
        pd.DataFrame: A DataFrame containing the results of all experiments.
    """
    with Pool(cpu_count()) as pool:
        all_results = []

        # Run experiments in parallel
        for result_set in tqdm(pool.imap_unordered(process_combination,
                                                   [(preprocessor, feature_extractor, n_folds)
                                                    for preprocessor, feature_extractor in
                                                    preprocessor_feature_combinations]),
                               total=len(preprocessor_feature_combinations), desc="Running experiments"):
            all_results.extend(result_set)

    # Convert results to DataFrame
    results_df = pd.DataFrame([r.__dict__ for r in all_results])

    # Plot the results
    plot_results(results_df)

    return results_df
