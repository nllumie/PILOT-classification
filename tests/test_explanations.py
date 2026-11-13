import unittest

import numpy as np
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.classification.explanations import (
    check_faithfulness, explain_logreg, explain_pilot_tree,
    explain_raffle, explain_random_forest, format_card,
)
from pilot_classification import PILOTClassifier, RaFFLEClassifier


def _data(seed=0, n=240):
    X, y = make_classification(n_samples=n, n_features=4, n_informative=3,
                               n_redundant=0, random_state=seed)
    return X, y


class ExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = _data()
        cls.rows = cls.X[:6]
        cls.logreg = Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler()),
                               ("model", LogisticRegression(max_iter=1000))]).fit(cls.X, cls.y)
        cls.rf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("model", RandomForestClassifier(n_estimators=15, min_samples_leaf=5,
                                                            random_state=0))]).fit(cls.X, cls.y)
        cls.raffle = Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("model", RaFFLEClassifier(n_estimators=5, max_depth=3,
                                                          max_features_tree=3,
                                                          min_samples_leaf=10,
                                                          random_state=0))]).fit(cls.X, cls.y)

    def test_logreg_reconstructs_decision_function(self):
        lr = self.logreg.named_steps["model"]
        for x in self.rows:
            e = explain_logreg(self.logreg, x)
            z = self.logreg[:-1].transform(x.reshape(1, -1))
            self.assertAlmostEqual(e.total(), float(lr.decision_function(z)[0]), places=10)
            self.assertTrue(check_faithfulness(self.logreg, x, e)["ok"])

    def test_random_forest_contributions_sum_to_probability(self):
        for x in self.rows:
            e = explain_random_forest(self.rf, x)
            p = self.rf.predict_proba(x.reshape(1, -1))[0, 1]
            self.assertAlmostEqual(e.total(), float(p), places=10)
            self.assertEqual(len(e.per_tree), 15)
            self.assertTrue(check_faithfulness(self.rf, x, e)["ok"])

    def test_pilot_tree_decomposition_is_exact(self):
        tree = PILOTClassifier(max_depth=3, min_samples_leaf=10, random_state=1).fit(self.X, self.y)
        for x in self.rows:
            _, _, logit_sum = explain_pilot_tree(tree, x)
            p = tree.predict_proba(x.reshape(1, -1))[0, 1]
            self.assertAlmostEqual(float(expit(logit_sum)), float(p), places=10)

    def test_raffle_matches_forest_probability_and_class(self):
        forest = self.raffle.named_steps["model"]
        for x in self.rows:
            e = explain_raffle(self.raffle, x)
            p = self.raffle.predict_proba(x.reshape(1, -1))[0, 1]
            self.assertAlmostEqual(e.probability_positive, float(p), places=10)
            self.assertEqual(e.predicted_class, self.raffle.predict(x.reshape(1, -1))[0])
            self.assertEqual(len(e.per_tree), len(forest.estimators_))
            # feature indices are global, within the input width
            self.assertTrue(all(0 <= j < self.X.shape[1] for j in e.contributions))
            self.assertFalse(e.exact)
            self.assertTrue(check_faithfulness(self.raffle, x, e)["ok"])

    def test_raffle_per_tree_sums_reproduce_each_tree(self):
        forest = self.raffle.named_steps["model"]
        x = self.rows[0]
        z = self.raffle[:-1].transform(x.reshape(1, -1))[0]
        for est, feats in zip(forest.estimators_, forest.estimators_features_):
            _, _, s = explain_pilot_tree(est, z[feats])
            p = est.predict_proba(z[feats].reshape(1, -1))[0, 1]
            self.assertAlmostEqual(float(expit(s)), float(p), places=10)

    def test_card_has_same_layout_for_every_model(self):
        names = ["f1", "f2", "f3", "f4"]
        classes = {0: "A", 1: "B"}
        x = self.rows[0]
        cards = [format_card(explain_logreg(self.logreg, x), x, names, classes),
                 format_card(explain_random_forest(self.rf, x), x, names, classes),
                 format_card(explain_raffle(self.raffle, x), x, names, classes)]
        for c in cards:
            self.assertTrue(c.startswith("Inputs:"))
            self.assertIn("Model prediction:", c)
            self.assertIn("Explanation:", c)
            self.assertIn("Combined result favours", c)
        self.assertIn("approximate", cards[2])
        self.assertNotIn("approximate", cards[0])

    def test_main_feature_is_largest_absolute_contribution(self):
        e = explain_logreg(self.logreg, self.rows[0])
        j = e.main_feature()
        self.assertEqual(abs(e.contributions[j]), max(abs(v) for v in e.contributions.values()))


if __name__ == "__main__":
    unittest.main()
