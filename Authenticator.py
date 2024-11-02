import numpy as np
from typing import List
from Classifier import Classifier

class Authenticator(Classifier):
    # Biometric authentication class. Verifies if the provided template matches
    # a registered user

    def __init__(self, registered_templates: List[np.ndarray], threshold: float):
        self.registered_templates = registered_templates
        self.threshold = threshold
        self.classifier = self.select_model(registered_templates)
        
        labels = np.zeros(len(registered_templates))  # Labels for one-class training
        self.classifier.train(registered_templates, labels)

    def classify(self, template: np.ndarray) -> bool:
        # Classifies a template by checking its match with registered templates
        # using the trained model
        prediction = self.classifier.predict_proba(template)
        return prediction >= self.threshold  # Return True if above threshold
