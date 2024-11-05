import pickle
import os

from Classifier import XGBoostClassifier
from DataSource import GetEcgIDData
from FeatureExtractor import StatisticalTimeExtractor, DiscreteCosineExtractor
from Preprocessor import BasicPreprocessor
from utils import train_test_split

# Define the cache file path
cache_file = 'data_cache.pkl'

# Load or process data
if os.path.exists(cache_file):
    print("Loading data from cache...")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
else:
    print("Loading data from source and caching it...")
    data = [person for person in GetEcgIDData('ecg-id-database-1.0.0', BasicPreprocessor(), DiscreteCosineExtractor())]
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
for person in test:
    print(person.uid, classifier.identify(person))
    print(person.uid, classifier.authenticate(person))
