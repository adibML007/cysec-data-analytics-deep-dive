import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve
from notebook import X, y, models
import numpy as np

class TestNotebookHelper(unittest.TestCase):

    def test_data_shapes(self):
        """Test if the data shapes are consistent."""
        self.assertEqual(X.shape[0], len(y), "Number of samples in X and y should match.")

    def test_confusion_matrix(self):
        """Test if confusion matrix computation works."""
        model = LogisticRegression(max_iter=200)
        model.fit(X.iloc[:100], y[:100])  # Train on a subset
        y_pred = model.predict(X.iloc[100:200])  # Predict on another subset
        cm = confusion_matrix(y[100:200], y_pred)
        self.assertEqual(cm.shape, (2, 2), "Confusion matrix should be 2x2 for binary classification.")

    def test_roc_curve(self):
        """Test if ROC curve computation works."""
        model = LogisticRegression(max_iter=200)
        model.fit(X.iloc[:100], y[:100])  # Train on a subset
        y_score = model.predict_proba(X.iloc[100:200])[:, 1]
        fpr, tpr, _ = roc_curve(y[100:200], y_score, pos_label='anomaly')
        self.assertTrue(len(fpr) > 0 and len(tpr) > 0, "ROC curve should return valid FPR and TPR arrays.")

    def test_random_forest_feature_importance(self):
        """Test if RandomForestClassifier computes feature importances."""
        model = RandomForestClassifier()
        model.fit(X, y)
        self.assertTrue(hasattr(model, "feature_importances_"), "RandomForestClassifier should have feature_importances_ attribute.")
        self.assertEqual(len(model.feature_importances_), X.shape[1], "Feature importances should match the number of features.")

if __name__ == "__main__":
    unittest.main()