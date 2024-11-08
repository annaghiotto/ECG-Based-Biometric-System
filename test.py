import pickle
import os

from Classifier import XGBoostClassifier
from DataSource import GetEcgIDData, GetSBData
from FeatureExtractor import StatisticalTimeExtractor, DiscreteCosineExtractor, PCAExtractor, SARModelExtractor, StatisticalTimeFreqExtractor
from Preprocessor import BasicPreprocessor, SARModelPreprocessor
from utils import train_test_split, k_fold_split

# Define the cache file path
cache_file = 'data_cache.pkl'

# Load or process data
if os.path.exists(cache_file):
    print("Loading data from cache...")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
else:
    print("Loading data from source and caching it...")
    # data = [person for person in GetSBData('SB_ECGDatabase_01', SARModelPreprocessor(), SARModelExtractor())]
    data = [person for person in GetEcgIDData('ecg-id-database-1.0.0', BasicPreprocessor(), StatisticalTimeFreqExtractor())]

    # Cache the data for future runs
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

# Perform the train-test split
train, test = train_test_split(data, 0.2)

# Initialize the classifier
classifier = XGBoostClassifier(threshold=0.5)

# Fit the classifier
classifier.fit(train, test)

# Test classifier predictions
correct_identifications = 0
successful_authentications = 0
for person in test:
    identified_uid = classifier.identify(person)
    is_authenticated = classifier.authenticate(person)

    # Check if identification is correct
    if identified_uid == person.uid:
        correct_identifications += 1

    # Check if authentication is correct
    if is_authenticated:
        successful_authentications += 1

    """
    print(person.uid, identified_uid)
    print(person.uid, is_authenticated)
    """

print(f"Correct identifications: {correct_identifications} out of {len(test)} ({(correct_identifications / len(test)) * 100:.2f}%)")
print(f"Successful authentications: {successful_authentications} out of {len(test)} ({(successful_authentications / len(test)) * 100:.2f}%)")


# KFold
k = 3  # Number of folds
folds = k_fold_split(data, k)

print("################# K-fold cross-validation, k=", k, " #################")

f = 1
for train, test in folds:
    print("####### Fold ", f, "/", k, "#######")
    f += 1
    # Initialize the classifier
    classifier = XGBoostClassifier(threshold=0.5)

    # Fit the classifier
    classifier.fit(train, test)

    # Test classifier predictions
    correct_identifications = 0
    successful_authentications = 0
    for person in test:
        identified_uid = classifier.identify(person)
        is_authenticated = classifier.authenticate(person)

        # Check if identification is correct
        if identified_uid == person.uid:
            correct_identifications += 1

        # Check if authentication is correct
        if is_authenticated:
            successful_authentications += 1

        """
        print(person.uid, identified_uid)
        print(person.uid, is_authenticated)
        """

    print(
        f"Correct identifications: {correct_identifications} out of {len(test)} ({(correct_identifications / len(test)) * 100:.2f}%)")
    print(
        f"Successful authentications: {successful_authentications} out of {len(test)} ({(successful_authentications / len(test)) * 100:.2f}%)")


