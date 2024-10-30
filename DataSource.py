import os
import wfdb
import numpy as np
from abc import abstractmethod, ABC
from dataclasses import dataclass


@dataclass
class Instance:
    signals: []
    id: int


class DataSource(ABC):
    def __init__(self, filename):
        self.person = 1
        self.filename = filename

    def __iter__(self):
        return self

    def __next__(self) -> Instance:
        pass


class GetSBData(DataSource):

    def __init__(self, filename):
        super().__init__(filename)
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

    def __next__(self) -> Instance:
        try:
            person = self.person
            self.person += 1
            return self.person_signals[person], person
        except KeyError:
            raise StopIteration


class GetEcgIDData(DataSource):

    def __next__(self) -> Instance:
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
                    return person_signals, person
        else:
            raise StopIteration


data = GetEcgIDData('ecg-id-database-1.0.0')

for i in data:
    print(i)

data_sb = GetSBData('SB_ECGDatabase_01')

for i in data_sb:
    print(i)