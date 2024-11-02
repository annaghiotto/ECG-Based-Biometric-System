from abc import ABC
from dataclasses import dataclass
from typing import List

import numpy as np

from Person import Person
import xgboost as xgb


@dataclass
class Classifier(ABC):
    threshold: float

    def fit(self, person_list: List[Person]):
        pass

    def identify(self, person: Person) -> int | None:
        pass

    def authenticate(self, person: Person) -> bool:
        pass


@dataclass
class XGBoostClassifier(Classifier):
    def __post_init__(self):
        self.model = xgb.XGBClassifier(objective="multi:softmax")

    def fit(self, person_list: List[Person]):
        X = [
            template
            for person in person_list
            for template in person.templates
        ]

        y = [person.uid for person in person_list for _ in person.templates]

        n_classes = np.unique(y).shape[0]
        self.model.n_classes_ = n_classes
        self.model.fit(X, y)

    def identify(self, person: Person) -> int | None:
        predicted_classes = []
        for template in person.templates:
            prediction_proba = self.model.predict_proba([template])[0]
            prediction = np.argmax(prediction_proba)
            max_proba = prediction_proba[prediction]

            if max_proba >= self.threshold:
                predicted_classes.append(prediction)

        if len(predicted_classes) == 0:
            return None

        return max(set(predicted_classes), key=predicted_classes.count)

    def authenticate(self, person: Person) -> bool:
        random_template_idx = np.random.randint(0, len(person.templates))
        random_template = person.templates[random_template_idx]
        prediction_proba = self.model.predict_proba([random_template])[0]
        return prediction_proba[person.uid] >= self.threshold
