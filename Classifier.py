from abc import ABC
from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

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
        y_pred = self.model.predict(eval_X)
        accuracy = accuracy_score(eval_y, y_pred)
        print(f"Accuracy: {accuracy}")

        y_true = []
        y_scores = []

        for person in eval_set:
            for template in person.templates_flat:
                prediction_proba = self.model.predict_proba([template])[0]

                # Append positive match
                y_true.append(1)
                y_scores.append(prediction_proba[person.uid])

                # Append synthetic negative match
                other_uid = (person.uid + 1) % self.model.n_classes_
                y_true.append(0)
                y_scores.append(prediction_proba[other_uid])

            # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        # Filter out zero or near-zero differences in fpr for stability
        fpr, tpr = np.array(fpr), np.array(tpr)
        unique_fpr_indices = np.where(np.diff(fpr) > 1e-6)[0]
        if len(unique_fpr_indices) < 2:
            print("Error: Insufficient unique FPR values to compute ROC curve.")
            return {"EER": None, "AUC": None}

        # Ensure no zero-difference in FPR for interpolation stability
        fpr, tpr = fpr[unique_fpr_indices], tpr[unique_fpr_indices]

        # Avoid NaN issues by trimming boundaries and interpolating safely
        try:
            # Interpolate EER
            interp_func = interp1d(fpr, tpr, fill_value="extrapolate", bounds_error=False)
            eer = brentq(lambda x: 1. - x - interp_func(x), 0., 1.)
        except ValueError as e:
            print(f"Error calculating EER: {e}")
            eer = None

        # Calculate AUC if possible
        roc_auc = auc(fpr, tpr) if len(fpr) > 1 and len(tpr) > 1 else None

        print(f"Equal Error Rate: {eer}")
        print(f"Area Under the Curve: {roc_auc}")

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
