from abc import ABC
from dataclasses import dataclass
from typing import List
import numpy as np
from Person import Person
import xgboost as xgb


@dataclass
class Classifier(ABC):
    threshold: float

    def fit(self, train: List[Person], eval_set: List[Person]):
        pass

    def identify(self, person: Person) -> int | None:
        pass

    def authenticate(self, person: Person) -> bool:
        pass


@dataclass
class XGBoostClassifier(Classifier):
    def __post_init__(self):
        self.model = xgb.XGBClassifier(objective="multi:softmax", verbosity=2)

    def fit(self, train: List[Person], eval_set: List[Person]):
        X = [
            template
            for person in train
            for template in person.templates_flat
        ]
        y = [person.uid for person in train for _ in person.templates_flat]

        eval_X = [
            template
            for person in eval_set
            for template in person.templates_flat
        ]

        eval_y = [person.uid for person in eval_set for _ in person.templates_flat]

        n_classes = np.unique(y).shape[0]
        self.model.n_classes_ = n_classes

        print(np.array(X).shape)

        # Fit model with evaluation and logging
        print("Starting model training with iteration logging:")
        self.model.fit(
            X, y,
            eval_set=[(eval_X, eval_y)],
            verbose=True
        )
        print("Model training complete.")

    def identify(self, person: Person) -> int | None:
        predicted_classes = []
        for template in person.templates_flat:
            prediction_proba = self.model.predict_proba([template])[0]
            prediction = np.argmax(prediction_proba)
            max_proba = prediction_proba[prediction]

            if max_proba >= self.threshold:
                predicted_classes.append(prediction)

        if len(predicted_classes) == 0:
            return None

        return max(set(predicted_classes), key=predicted_classes.count)

    def authenticate(self, person: Person) -> bool:
        random_template_idx = np.random.randint(0, len(person.templates_flat))
        random_template = person.templates_flat[random_template_idx]
        prediction_proba = self.model.predict_proba([random_template])[0]
        return prediction_proba[person.uid] >= self.threshold