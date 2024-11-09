import pickle
import os

from Classifier import XGBoostClassifier
from DataSource import GetEcgIDData
from FeatureExtractor import StatisticalTimeExtractor
from Preprocessor import BasicPreprocessor
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
    data = [person for person in GetEcgIDData('ecg-id-database-1.0.0', BasicPreprocessor(), StatisticalTimeExtractor())]

    # Cache the data for future runs
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

# Perform the train-test split
train, test = train_test_split(data, 0.2)

# Initialize the classifier
classifier = XGBoostClassifier(threshold=0.5)

# Fit the classifier
classifier.fit(train, train)

accuracy, eer, eer_threshold, auc = classifier.evaluate(test)
print(f"Accuracy: {accuracy}")
if eer is not None:
    print(f"EER: {eer}")
    print(f"EER Threshold: {eer_threshold}")
if auc is not None:
    print(f"AUC: {auc}")

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

accuracy_sum = 0.0
eer_sum = 0.0
eer_threshold_sum = 0.0
auc_sum = 0.0
correct_identifications = 0
successful_authentications = 0
eer_err = False
auc_err = False
f = 1
for train, test in folds:
    print("####### Fold ", f, "/", k, "#######")
    f += 1
    # Initialize the classifier
    classifier = XGBoostClassifier(threshold=0.5)

    # Fit the classifier
    classifier.fit(train, train)

    accuracy, eer, eer_threshold, auc = classifier.evaluate(test)
    accuracy_sum += accuracy
    if eer is not None:
        eer_sum += eer
    else:
        eer_err = True
    if eer_threshold is not None:
        eer_threshold_sum += eer_threshold
    else:
        eer_err = True
    if auc is not None:
        auc_sum += auc
    else:
        auc_err = True


    # Test classifier predictions
    for person in test:
        identified_uid = classifier.identify(person)
        is_authenticated = classifier.authenticate(person)

        # Check if identification is correct
        if identified_uid == person.uid:
            correct_identifications += 1

        # Check if authentication is correct
        if is_authenticated:
            successful_authentications += 1

print(
    f"Correct identifications: {correct_identifications} out of {len(test)*k} ({(correct_identifications / (len(test)*k)) * 100:.2f}%)")
print(
        f"Successful authentications: {successful_authentications} out of {len(test)*k} ({(successful_authentications / (len(test)*k)) * 100:.2f}%)")

auc = auc_sum/k
accuracy = accuracy_sum/k
print(f"Accuracy: {accuracy}")
if eer_err is False:
    eer = eer_sum/k
    print(f"EER: {eer}")
    eer_threshold_sum = eer_threshold/k
    print(f"EER Threshold: {eer_threshold}")
if auc_err is False:
    auc = auc_sum/k
    print(f"AUC: {auc}")
    