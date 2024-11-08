import math
from dataclasses import dataclass

from sklearn.model_selection import KFold

from custom_types import Signal, Template, Features
from Templates import TemplatesFactory
from typing import List, Generator, Tuple
from itertools import chain


@dataclass
class Person:
    templates: List["Template"]
    uid: int

    @property
    def templates_flat(self) -> List["Features"]:
        return list(chain(*self.templates))

    def train_test_split(self, test_size: float) -> ("Person", "Person"):
        n_templates = len(self.templates)
        if n_templates == 1:
            n_templates = len(self.templates[0])
            n_test = math.ceil(n_templates * test_size)
            return Person([self.templates[0][n_test:]], self.uid), Person([self.templates[0][:n_test]], self.uid)

        n_test = math.ceil(n_templates * test_size)
        return Person(self.templates[n_test:], self.uid), Person(self.templates[:n_test], self.uid)

    def k_fold_split(self, k: int) -> List[Tuple["Person", "Person"]]:
        """Perform K-Fold split on the templates and return train/test splits for each fold."""


        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        folds = []
        n_templates = len(self.templates)
        # print(n_templates)
        # print(k)

        if n_templates < k:
            print("Number of templates must be at least equal to k for KFold splitting.")
            train_fold = [[] for _ in range(k)]
            test_fold = [[] for _ in range(k)]
            for j in range(n_templates):
                i = 0
                for train_idx, test_idx in kf.split(self.templates[j]):
                    # print(train_idx)
                    # for i in train_idx:
                        # print(self.templates[j][i])
                    train_fold[i].append([self.templates[j][i] for i in train_idx])
                    test_fold[i].append([self.templates[j][i] for i in test_idx])
                    i += 1
                    # print(train_templates)

            for i in range(k):
                folds.append(
                    (Person(train_fold[i], self.uid), Person(test_fold[i], self.uid))
                )

            print("len(folds) n_templates < k: ", len(folds))
            return folds
        else:
            for train_idx, test_idx in kf.split(self.templates):
                train_templates = [self.templates[i] for i in train_idx]
                test_templates = [self.templates[i] for i in test_idx]
                folds.append(
                    (Person(train_templates, self.uid), Person(test_templates, self.uid))
                )
            print("len(folds): ", len(folds))
            return folds

@dataclass
class PersonFactory:
    templates_factory: "TemplatesFactory"

    def create(self, signals: List[Signal], uid: int) -> Person:
        return Person(self.templates_factory.from_signals(signals), uid)
