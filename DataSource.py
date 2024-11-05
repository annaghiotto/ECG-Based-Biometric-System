import os
import wfdb
import numpy as np
from abc import ABC
from Person import PersonFactory, Person


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
        self.fs = 1000
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

    def __next__(self) -> Person:
        try:
            person = self.person
            self.person += 1
            return self.person_factory.create(self.person_signals[person], person-1, self.fs)
        except KeyError:
            raise StopIteration


class GetEcgIDData(DataSource):

    def __next__(self) -> Person:
        self.fs = 500
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
                    return self.person_factory.create(person_signals, person-1, self.fs)
        else:
            raise StopIteration
