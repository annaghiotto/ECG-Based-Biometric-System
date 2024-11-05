import math
from dataclasses import dataclass

from custom_types import Signal, Template, Features
from Templates import TemplatesFactory
from typing import List
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


@dataclass
class PersonFactory:
    templates_factory: "TemplatesFactory"

    def create(self, signals: List[Signal], uid: int) -> Person:
        return Person(self.templates_factory.from_signals(signals), uid)
