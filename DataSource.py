import os
import wfdb
import numpy as np
from abc import ABC

from FeatureExtractor import FeatureExtractor
from Person import PersonFactory, Person
from Preprocessor import Preprocessor
from Templates import TemplatesFactory


class DataSource(ABC):
    def __init__(self, filename, person_factory: PersonFactory):
        self.person = 1
        self.filename = filename
        self.person_factory = person_factory

    def __iter__(self):
        return self

    def __next__(self) -> Person:
        pass


class GetSBData(DataSource):

    def __init__(self, filename, person_factory: PersonFactory):
        super().__init__(filename, person_factory)
        self.person_signals = {}
        filelist = iter(os.listdir(os.fsencode(self.filename)))
        for filename in filelist:
            filename = self.filename + '/' + os.fsdecode(filename)
            print(filename)
            signal = np.loadtxt(filename)
            person = int(filename.split('_')[4].split('u')[1].split('.')[0])
            try:
                self.person_signals[person].append(signal)
            except:
                self.person_signals[person] = [signal]

    def __next__(self) -> Person:
        try:
            person = self.person
            self.person += 1
            return self.person_factory.create(self.person_signals[person], person)
        except KeyError:
            raise StopIteration


class GetEcgIDData(DataSource):

    def __next__(self) -> Person:
        record = 1
        person_signals = []
        if os.path.exists(self.filename + '/Person_' + f"{self.person:02}"):
            while True:
                try:
                    filename = self.filename + '/Person_' + f"{self.person:02}" + '/rec_' + str(record)
                    signal, fields = wfdb.rdsamp(filename)
                    person_signals.append(signal)
                    record += 1
                except FileNotFoundError:
                    person = self.person
                    self.person += 1
                    return self.person_factory.create(person_signals, person)
        else:
            raise StopIteration


template_factory = TemplatesFactory(Preprocessor(), FeatureExtractor())
person_factory = PersonFactory(template_factory)
data = GetEcgIDData('ecg-id-database-1.0.0', person_factory)

for i in data:
    print(i)

data_sb = GetSBData('SB_ECGDatabase_01', person_factory)

for i in data_sb:
    print(i)
