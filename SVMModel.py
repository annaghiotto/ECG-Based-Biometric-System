from sklearn.svm import SVC
import numpy as np


class SVMModel:
    def __init__(self, probability: bool = True):
        # Initialize the SVM classifier model with probability support
        self.model = SVC(probability=probability)

    def train(self, templates: np.ndarray, labels: np.ndarray):
        # Train the SVM model with the provided templates and labels
        self.model.fit(templates, labels)

    def predict(self, template: np.ndarray) -> int:
        # Predict the class label for a given template
        return int(self.model.predict([template])[0])

    def predict_proba(self, template: np.ndarray) -> float:
        # Predict the probability of belonging to the class
        return self.model.predict_proba([template])[0][0]
