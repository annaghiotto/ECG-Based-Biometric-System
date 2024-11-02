from dataclasses import dataclass

from Signal import Signal
from Templates import Templates, TemplatesFactory
from typing import List


@dataclass
class Person:
    templates: Templates
    uid: int


@dataclass
class PersonFactory:
    templates_factory: "TemplatesFactory"

    def create(self, signals: List[Signal], uid: int) -> Person:
        return Person(self.templates_factory.from_signals(signals), uid)
