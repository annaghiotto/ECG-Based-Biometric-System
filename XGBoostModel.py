from xgboost import XGBClassifier
import numpy as np

class XGBoostModel:
    def __init__(self):
        # Initialize the XGBoost classifier model
        self.model = XGBClassifier()

    def train(self, templates: np.ndarray, labels: np.ndarray):
        # Train the XGBoost model with the provided templates and labels
        self.model.fit(templates, labels)

    def predict(self, template: np.ndarray) -> int:
        # Predict the class label for a given template
        return int(self.model.predict([template])[0])

    def predict_proba(self, template: np.ndarray) -> float:
        # Predict the probability of belonging to the class
        return self.model.predict_proba([template])[0][0]
