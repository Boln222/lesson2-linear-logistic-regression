"""Unit-тесты критических функций. Запуск: python test_homework.py

Метрики сверяются с готовыми из sklearn: если наша формула ошибается, тест упадёт.
"""
import unittest

import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics as skm
from torch.utils.data import DataLoader, TensorDataset

from homework_datasets import CSVDataset
from homework_experiments import add_group_statistics, add_interaction_features, add_polynomial_features
from homework_model_modification import (EarlyStopping, LinearRegressionModel, classification_metrics,
                                         regularization_penalty, train_model)


class TestTraining(unittest.TestCase):

    def test_penalty(self):
        """L1 = λ·Σ|w|, L2 = λ·Σw², bias не входит."""
        model = LinearRegressionModel(2)
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[1.0, -2.0]]))
            model.linear.bias.fill_(100.0)
        self.assertAlmostEqual(regularization_penalty(model, l1=0.5).item(), 1.5)  # 0.5 * (1 + 2)
        self.assertAlmostEqual(regularization_penalty(model, l2=0.1).item(), 0.5)  # 0.1 * (1 + 4)

    def test_early_stopping(self):
        """Loss лучший на эпохе 3, дальше хуже: при patience=2 стоп на эпохе 5, веса эпохи 3."""
        model = LinearRegressionModel(1)
        stopper = EarlyStopping(patience=2, min_delta=0)
        for epoch, loss in enumerate([1.0, 0.8, 0.5, 0.6, 0.7], start=1):
            with torch.no_grad():
                model.linear.weight.fill_(epoch)
            stopped = stopper.step(loss, model, epoch)
        self.assertTrue(stopped)
        self.assertEqual(stopper.best_epoch, 3)
        stopper.restore_best(model)
        self.assertEqual(model.linear.weight.item(), 3.0)

    def test_learns_line(self):
        """На данных y = 2x + 1 модель находит w ≈ 2, b ≈ 1."""
        torch.manual_seed(0)
        X = torch.rand(200, 1)
        loader = DataLoader(TensorDataset(X, 2 * X[:, 0] + 1), batch_size=32, shuffle=True)
        model = LinearRegressionModel(1)
        train_model(model, loader, loader, nn.MSELoss(), epochs=300, lr=0.1, log_every=0)
        self.assertAlmostEqual(model.linear.weight.item(), 2.0, delta=0.05)
        self.assertAlmostEqual(model.linear.bias.item(), 1.0, delta=0.05)


class TestMetrics(unittest.TestCase):

    def test_multiclass_vs_sklearn(self):
        torch.manual_seed(1)
        y = torch.tensor([0, 1, 2, 2, 1, 0, 2, 1, 1, 0, 2, 2])
        probs = torch.softmax(torch.randn(12, 3), dim=1)
        pred = probs.argmax(1)
        m = classification_metrics(y, probs)
        self.assertEqual(m["confusion_matrix"].tolist(), skm.confusion_matrix(y, pred).tolist())
        self.assertAlmostEqual(m["precision"], skm.precision_score(y, pred, average="macro", zero_division=0), 5)
        self.assertAlmostEqual(m["recall"], skm.recall_score(y, pred, average="macro"), 5)
        self.assertAlmostEqual(m["f1"], skm.f1_score(y, pred, average="macro"), 5)
        self.assertAlmostEqual(m["roc_auc"], skm.roc_auc_score(y, probs, multi_class="ovr"), 5)

    def test_binary_vs_sklearn(self):
        y = torch.tensor([0, 0, 1, 1, 1, 0, 1, 0])
        p1 = torch.tensor([0.1, 0.6, 0.8, 0.4, 0.9, 0.3, 0.7, 0.2])
        pred = (p1 > 0.5).long()
        m = classification_metrics(y, torch.stack([1 - p1, p1], dim=1))
        self.assertAlmostEqual(m["precision"], skm.precision_score(y, pred), 5)
        self.assertAlmostEqual(m["recall"], skm.recall_score(y, pred), 5)
        self.assertAlmostEqual(m["f1"], skm.f1_score(y, pred), 5)
        self.assertAlmostEqual(m["roc_auc"], skm.roc_auc_score(y, p1), 5)


class TestData(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({"age": [20, 30, None, 40, 50, 60],             # число с пропуском
                                "smoker": ["yes", "no", "no", "yes", "no", None],  # бинарный с пропуском
                                "city": ["A", "B", "C", "A", "B", "C"],         # категория
                                "price": [10.0, 20, 30, 40, 50, 60]})           # ответ

    def test_preprocessing(self):
        ds = CSVDataset(self.df, "price")
        self.assertEqual(ds.feature_names, ["age", "smoker=yes", "city=A", "city=B", "city=C"])
        self.assertFalse(torch.isnan(ds.X).any())                       # пропуски заполнены
        self.assertAlmostEqual(ds.X[:, 0].mean().item(), 0, places=5)   # стандартизация
        self.assertEqual(ds.X[:, 1].tolist(), [1, 0, 0, 1, 0, 0])       # yes -> 1, пропуск -> мода "no"
        self.assertEqual(ds.X[0, 2:].tolist(), [1, 0, 0])               # one-hot города A
        x, y = ds[0]                                                    # __getitem__
        self.assertEqual(len(ds), 6)                                    # __len__

    def test_val_uses_train_statistics(self):
        train_ds = CSVDataset(self.df.iloc[:4], "price")
        val_ds = CSVDataset(self.df.iloc[4:], "price", reference=train_ds)
        rule = train_ds.preprocessor.rules["age"]
        self.assertAlmostEqual(val_ds.X[0, 0].item(), (50 - rule["mean"]) / rule["std"], places=5)
        self.assertAlmostEqual(val_ds.inverse_target(val_ds.y)[0].item(), 50.0, places=4)

    def test_feature_engineering(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "smoker": ["no", "yes", "yes"], "g": ["a", "a", "b"]})
        self.assertEqual(add_polynomial_features(df, ["x"])["x^2"].tolist(), [1, 4, 9])
        self.assertEqual(add_interaction_features(df, [("x", "smoker")])["x*smoker"].tolist(), [0, 2, 3])
        test = pd.DataFrame({"x": [10.0], "smoker": ["no"], "g": ["a"]})
        _, new_test = add_group_statistics(df.iloc[:2], [test], "g", "x")
        self.assertEqual(new_test["x_mean_by_g"].iloc[0], 1.5)  # среднее по train, а не 10


if __name__ == "__main__":
    unittest.main(verbosity=2)
