import os
import wfdb
import numpy as np

from FeatureExtractor import FeatureExtractor
from Person import PersonFactory
from Preprocessor import Preprocessor
from Templates import TemplatesFactory
from TqdmIteratorBase import TqdmIteratorBase


class GetSBData(TqdmIteratorBase):
    """
    Data source for SB Data.
    Iterates over persons and yields Person objects with their signals.
    """

    def __init__(self, filename, preprocessor: Preprocessor, feature_extractor: FeatureExtractor, desc='GetSBData',
                 **tqdm_kwargs):
        """
        Initializes the GetSBData iterator.

        :param filename: Directory containing SB data files.
        :param preprocessor: Preprocessor instance.
        :param feature_extractor: FeatureExtractor instance.
        :param desc: Description for the tqdm progress bar.
        :param tqdm_kwargs: Additional keyword arguments for tqdm.
        """
        # Initialize person_signals
        self.person_signals = {}
        self.filename = filename

        # Process files to populate person_signals
        filelist = os.listdir(self.filename)
        for file in filelist:
            filepath = os.path.join(self.filename, file)
            if os.path.isfile(filepath):
                # print(f"Processing file: {filepath}")
                signal = np.loadtxt(filepath)
                try:
                    # Extract person number from filename
                    # Assumes filename format contains '_u<person_number>.'
                    person_str = file.split('_')[2]
                    person_number = int(person_str.split('u')[1].split('.')[0])
                    # Append the signal (assuming second column)
                    self.person_signals.setdefault(person_number, []).append(signal[:, 1])
                except (IndexError, ValueError) as e:
                    print(f"Skipping file {filepath} due to parsing error: {e}")

        # Determine total number of persons
        total_persons = len(self.person_signals)

        super().__init__(desc=desc, total=total_persons, **tqdm_kwargs)

        # Initialize other attributes
        self.person = 1
        self.person_factory = PersonFactory(
            TemplatesFactory(preprocessor, feature_extractor)
        )

    @property
    def fs(self) -> float:
        """
        Sampling frequency for SB data.

        :return: Sampling frequency.
        """
        return 1000

    def generator(self):
        """
        Generator that yields Person objects.
        """
        sorted_persons = sorted(self.person_signals.keys())
        for person in sorted_persons:
            signals = self.person_signals[person]
            yield self.person_factory.create(signals, person - 1)


class GetEcgIDData(TqdmIteratorBase):
    """
    Data source for ECG-ID Data.
    Iterates over persons and yields Person objects with their records.
    """

    def __init__(self, filename, preprocessor: Preprocessor, feature_extractor: FeatureExtractor, desc='GetEcgIDData',
                 **tqdm_kwargs):
        """
        Initializes the GetEcgIDData iterator.

        :param filename: Directory containing ECG-ID data.
        :param preprocessor: Preprocessor instance.
        :param feature_extractor: FeatureExtractor instance.
        :param desc: Description for the tqdm progress bar.
        :param tqdm_kwargs: Additional keyword arguments for tqdm.
        """
        self.filename = filename
        self.person_factory = PersonFactory(
            TemplatesFactory(preprocessor, feature_extractor)
        )

        # Determine total number of persons
        self.person_dirs = [
            d for d in os.listdir(self.filename)
            if os.path.isdir(os.path.join(self.filename, d)) and d.startswith('Person_')
        ]
        total_persons = len(self.person_dirs)

        super().__init__(desc=desc, total=total_persons, **tqdm_kwargs)

    @property
    def fs(self) -> float:
        """
        Sampling frequency for ECG-ID data.

        :return: Sampling frequency.
        """
        return 500

    def generator(self):
        """
        Generator that yields Person objects.
        """
        for person_dir in sorted(self.person_dirs):
            person_path = os.path.join(self.filename, person_dir)
            person_number_str = person_dir.split('_')[-1]
            try:
                person_number = int(person_number_str)
            except ValueError:
                print(f"Invalid person directory name: {person_dir}. Skipping.")
                continue

            person_signals = []
            record = 1
            while True:
                record_filename = os.path.join(person_path, f"rec_{record}")
                try:
                    signal, fields = wfdb.rdsamp(record_filename)
                    # Assuming the first column contains the desired signal
                    person_signals.append(signal[:, 0])
                    record += 1
                except FileNotFoundError:
                    if person_signals:
                        yield self.person_factory.create(person_signals, person_number - 1)
                    else:
                        print(f"No records found for {person_dir}. Skipping.")
                    break
                except Exception as e:
                    print(f"Error reading {record_filename}: {e}. Skipping this record.")
                    record += 1
