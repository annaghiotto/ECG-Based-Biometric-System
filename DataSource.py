import os
import wfdb
import numpy as np
from abc import ABC, abstractmethod

from FeatureExtractor import FeatureExtractor
from Person import PersonFactory, Person
from Preprocessor import Preprocessor
from Templates import TemplatesFactory


class DataSource(ABC):
    def __init__(self, filename, preprocessor: Preprocessor, feature_extractor: FeatureExtractor):
        self.person = 1
        self.filename = filename
        preprocessor.fs = self.fs
        self.person_factory = PersonFactory(
            TemplatesFactory(preprocessor, feature_extractor)
        )

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> Person:
        pass

    @property
    @abstractmethod
    def fs(self) -> float:
        pass


class GetSBData(DataSource):

    def __init__(self, filename, preprocessor: Preprocessor, feature_extractor: FeatureExtractor):
        super().__init__(filename, preprocessor, feature_extractor)

        self.person_signals = {}
        filelist = iter(os.listdir(os.fsencode(self.filename)))
        for filename in filelist:
            filename = self.filename + '/' + os.fsdecode(filename)
            print(filename)
            signal = np.loadtxt(filename)
            person = int(filename.split('_')[4].split('u')[1].split('.')[0])
            try:
                self.person_signals[person].append(signal[:, 1])
            except:
                self.person_signals[person] = [signal[:, 1]]

    @property
    def fs(self) -> float:
        return 1000

    def __next__(self) -> Person:
        try:
            person = self.person
            self.person += 1
            return self.person_factory.create(self.person_signals[person], person-1)
        except KeyError:
            raise StopIteration


class GetEcgIDData(DataSource):

    @property
    def fs(self) -> float:
        return 500

    def __next__(self) -> Person:
        record = 1
        person_signals = []
        if os.path.exists(self.filename + '/Person_' + f"{self.person:02}"):
            while True:
                try:
                    filename = self.filename + '/Person_' + f"{self.person:02}" + '/rec_' + str(record)
                    signal, fields = wfdb.rdsamp(filename)
                    person_signals.append(signal[:, 0])
                    record += 1
                except FileNotFoundError:
                    person = self.person
                    self.person += 1
                    return self.person_factory.create(person_signals, person-1)
        else:
            raise StopIteration
