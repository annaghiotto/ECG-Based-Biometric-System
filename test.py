from DataSource import GetEcgIDData, GetSBData
from FeatureExtractor import SimpleFeatureExtractor
from Person import PersonFactory
from Preprocessor import SimplePreprocessor, BasicPreprocessor
from Templates import TemplatesFactory
from Classifier import XGBoostClassifier

template_factory = TemplatesFactory(BasicPreprocessor(), SimpleFeatureExtractor())
person_factory = PersonFactory(template_factory)
data = [person for person in GetEcgIDData('ecg-id-database-1.0.0', person_factory)]

data_sb = [person for person in GetSBData('SB_ECGDatabase_01', person_factory)]

for person in data_sb:
    print(person)

classifier = XGBoostClassifier(
    threshold=0.5
)

classifier.fit(data_sb)

for person in data_sb:
    print(person.uid, classifier.identify(person))
    print(person.uid, classifier.authenticate(person))

#
# for i in data:
#     print(i)
#

#
# for i in data_sb:
#     print(i)
