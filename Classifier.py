from abc import ABC
from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
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

                if prediction_proba[person.uid] >= self.threshold:
                    y_true.append(1)
                    y_scores.append(prediction_proba[person.uid])

                elif any(element >= self.threshold for element in prediction_proba):
                    for element in prediction_proba:
                        if element >= self.threshold:
                            y_true.append(0)
                            y_scores.append(element)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        # Filter out zero or near-zero differences in fpr for stability
        fpr, tpr = np.array(fpr), np.array(tpr)
        unique_fpr_indices = np.where(np.diff(fpr) > 1e-6)[0]
        if len(unique_fpr_indices) < 2:
            print("Error: Insufficient unique FPR values to compute ROC curve.")
        else:
            fnr = 1 - tpr
            # Find the threshold where the difference between FPR and FNR is smallest
            eer_index = np.nanargmin(np.abs(fpr - fnr))
            eer_threshold = thresholds[eer_index]
            eer = fpr[eer_index]

            print(f"EER: {eer}")
            print(f"EER Threshold: {eer_threshold}")
        try:
            auc = roc_auc_score(y_true, y_scores)
            print(f"AUC: {auc}")
        except ValueError:
            print("Error: Insufficient unique FPR values to compute ROC curve.")


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