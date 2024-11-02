# identifier.py
import numpy as np
from typing import List
from Person import Person
from Classifier import Classifier

class Identifier(Classifier):
    # Biometric identification class. Identifies the template as belonging to one
    # of the registered users and returns the user ID

    def __init__(self, persons: List[Person]):
        self.persons = persons
        templates = [person.templates for person in persons]
        self.classifier = self.select_model(templates)

        training_templates = []
        training_labels = []
        for person in persons:
            training_templates.extend(person.templates)
            training_labels.extend([person.uid] * len(person.templates))

        self.classifier.train(training_templates, training_labels)

    def classify(self, template: np.ndarray) -> int:
        # Identifies the template by comparing it to registered templates and
        # returning the ID of the matching person
        return self.classifier.predict(template)
