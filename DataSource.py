import os
import wfdb
import numpy as np
from abc import abstractmethod, ABC
from dataclasses import dataclass

@dataclass
class Instance:
    signal: np.ndarray
    id: int


class DataSource(ABC):
    def __init__(self, filename):
        self.filename = filename
        self.filelist = []
        self.person = 1
        self.record = 1

    def __iter__(self):
        return self
    
    def __next__(self) -> Instance:
        pass
    

class GetSBData(DataSource):

    def __iter__(self):
        filelist = iter(os.listdir(os.fsencode(self.filename)))
        self.person = 0
        for filename in filelist:
            self.filelist.append(os.fsdecode(filename))
        return self

    def __next__(self) -> Instance:
        try:    
            filename = self.filename + '/' + self.filelist[self.person]
            print(filename)
            signals = np.loadtxt(filename)
            person = int(filename.split('_')[4].split('u')[1].split('.')[0])
            self.person += 1
            return signals, person
        except IndexError:
            raise StopIteration
    
class GetEcgIDData(DataSource):
    
    def __next__(self) -> Instance:
        filename = self.filename + '/Person_' + f"{self.person:02}" + '/rec_' + str(self.record)
        try:
            signals, fields = wfdb.rdsamp(filename)
            self.record += 1
            return signals, self.person
        except FileNotFoundError:
            self.person += 1
            self.record = 1
            filename = self.filename + '/Person_' + f"{self.person:02}" + '/rec_' + str(self.record)
            try:
                signals, fields = wfdb.rdsamp(filename)
                self.record += 1
                return signals, self.person
            except FileNotFoundError:
                raise StopIteration


data = GetEcgIDData('ecg-id-database-1.0.0')

for i in data:
    print(i)

data_sb = GetSBData('SB_ECGDatabase_01')

for i in data_sb:
    print(i)
