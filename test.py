from DataSource import GetEcgIDData, GetSBData
from FeatureExtractor import SimpleFeatureExtractor, StatisticalTimeExtractor
from Preprocessor import BasicPreprocessor
from Classifier import XGBoostClassifier

data = [person for person in GetEcgIDData('ecg-id-database-1.0.0', BasicPreprocessor(), StatisticalTimeExtractor())]

data_sb = [person for person in GetSBData('SB_ECGDatabase_01', BasicPreprocessor(), SimpleFeatureExtractor())]
#
# for person in data_sb:
#     print(person)

classifier = XGBoostClassifier(
    threshold=0.5
)

classifier.fit(data)

for person in data:
    print(person.uid, classifier.identify(person))
    print(person.uid, classifier.authenticate(person))