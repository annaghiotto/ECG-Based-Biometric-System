from DataSource import GetEcgIDData, GetSBData
from FeatureExtractor import SimpleFeatureExtractor
from Person import PersonFactory
from Preprocessor import SimplePreprocessor
from Templates import TemplatesFactory

template_factory = TemplatesFactory(SimplePreprocessor(), SimpleFeatureExtractor())
person_factory = PersonFactory(template_factory)
data = GetEcgIDData('ecg-id-database-1.0.0', person_factory)

for i in data:
    print(i)

data_sb = GetSBData('SB_ECGDatabase_01', person_factory)

for i in data_sb:
    print(i)
