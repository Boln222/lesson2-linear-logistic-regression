"""Задание 2.
2.1 Свой Dataset для CSV: загрузка, пропуски, нормализация, кодирование категорий.
2.2 Обучение на CSV: регрессия на data/insurance.csv (сумма страховки),
    бинарная классификация на data/titanic.csv (выжил ли пассажир).
    Модели и цикл обучения — из задания 1.
"""
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from homework_model_modification import (
    MODELS, PLOTS, EarlyStopping, LinearRegressionModel, LogisticRegressionModel,
    classification_metrics, logger, plot_confusion_matrix, plot_curves, predict,
    regression_metrics, setup_logging, train_model)

DATA = Path(__file__).parent / "data"


# ================= 2.1 Предобработка и Dataset =================

def detect_column_types(df, max_categories=20):
    types = {}
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique == 2:
            types[col] = "binary"
        elif pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        elif n_unique <= max_categories:
            types[col] = "categorical"
        else: 
            logger.warning("Столбец %s пропущен: %d разных значений", col, n_unique)
    return types


class TabularPreprocessor:
    def fit(self, df):        
        self.rules = {}
        for col, kind in detect_column_types(df).items():
            if kind == "numeric":
                fill = df[col].median()                      # пропуски -> медиана
                filled = df[col].fillna(fill)
                self.rules[col] = {"kind": kind, "fill": fill, "mean": filled.mean(), "std": filled.std()}
            else:
                values = sorted(df[col].dropna().unique())
                self.rules[col] = {"kind": kind, "fill": df[col].mode()[0], "values": values}  # пропуски -> мода
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for col, rule in self.rules.items():
            values = df[col].fillna(rule["fill"])
            if rule["kind"] == "numeric":
                out[col] = (values - rule["mean"]) / rule["std"]          # стандартизация
            elif rule["kind"] == "binary":
                one = rule["values"][-1]                                
                out[f"{col}={one}"] = (values == one).astype(float)
            else:
                for category in rule["values"]:                            # one-hot
                    out[f"{col}={category}"] = (values == category).astype(float)
        return out


class CSVDataset(Dataset):
    def __init__(self, data, target, task="regression", drop=(), reference=None):
        df = data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(data)
        df = df.drop(columns=list(drop)).dropna(subset=[target])
        features, y = df.drop(columns=[target]), df[target]
        self.task = task

        if reference is None:  # train: считаем всё сами
            self.preprocessor = TabularPreprocessor().fit(features)
            self.classes = sorted(y.unique()) if task == "classification" else None
            self.y_mean, self.y_std = (y.mean(), y.std()) if task == "regression" else (0, 1)
        else:                  # val/test: берём у train
            self.preprocessor = reference.preprocessor
            self.classes, self.y_mean, self.y_std = reference.classes, reference.y_mean, reference.y_std

        frame = self.preprocessor.transform(features)
        self.feature_names = list(frame.columns)
        self.X = torch.tensor(frame.to_numpy("float32"))
        if task == "classification":  # метки -> номера классов 0, 1, ... (нужно CrossEntropyLoss)
            self.y = torch.tensor(y.map({c: i for i, c in enumerate(self.classes)}).to_numpy(), dtype=torch.long)
        else:                          # ответ регрессии тоже стандартизуем
            self.y = torch.tensor(((y - self.y_mean) / self.y_std).to_numpy("float32"))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def inverse_target(self, y):
        return y * self.y_std + self.y_mean


def split_dataframe(df, stratify_column=None, seed=42):  # делим таблицу на три части (train/val/test)
    rest, test = train_test_split(df, test_size=0.2, random_state=seed,
                                  stratify=df[stratify_column] if stratify_column else None)
    train, val = train_test_split(rest, test_size=0.25, random_state=seed,
                                  stratify=rest[stratify_column] if stratify_column else None)
    return train, val, test


def make_datasets(df, target, task="regression", drop=()):
    train, val, test = split_dataframe(df, target if task == "classification" else None)
    train_ds = CSVDataset(train, target, task, drop)
    return (train_ds, CSVDataset(val, target, task, drop, reference=train_ds),
            CSVDataset(test, target, task, drop, reference=train_ds))


def regression_metrics_in_dollars(model, ds):
    return regression_metrics(ds.inverse_target(ds.y), ds.inverse_target(predict(model, ds.X)))


# ================= 2.2 Обучение на CSV =================

def run_regression_on_csv():
    """Линейная регрессия: сумма страховки по возрасту, ИМТ, курению и т.д."""
    train_ds, val_ds, test_ds = make_datasets(pd.read_csv(DATA / "insurance.csv"), "charges")
    print("\ninsurance.csv, признаки после предобработки:", train_ds.feature_names)

    torch.manual_seed(42)
    model = LinearRegressionModel(train_ds.X.shape[1])
    history = train_model(model, DataLoader(train_ds, batch_size=32, shuffle=True),
                          DataLoader(val_ds, batch_size=256), nn.MSELoss(), epochs=300, lr=0.01,
                          l2=0.001, early_stopping=EarlyStopping(patience=20))

    metrics = regression_metrics_in_dollars(model, test_ds)
    print(f"2.2 Линейная регрессия, тест: R2 = {metrics['r2']:.4f}, "
          f"RMSE = {metrics['rmse']:.0f} $, MAE = {metrics['mae']:.0f} $")
    weights = pd.Series(model.linear.weight.detach()[0].numpy(), index=train_ds.feature_names)
    print("Веса (признаки стандартизованы, поэтому их можно сравнивать):")
    print(weights.sort_values(key=abs, ascending=False).round(3).to_string())
    plot_curves(history, PLOTS / "2_2_loss_insurance.png", "2.2 Loss (insurance)")
    torch.save(model.state_dict(), MODELS / "linreg_insurance.pt")


def run_classification_on_csv():
    """Логистическая регрессия: выживет ли пассажир Титаника."""
    drop = ["PassengerId", "Name", "Ticket", "Cabin"]  # идентификаторы; Cabin пуст у 77%
    train_ds, val_ds, test_ds = make_datasets(pd.read_csv(DATA / "titanic.csv"), "Survived",
                                              task="classification", drop=drop)
    print("\ntitanic.csv, признаки после предобработки:", train_ds.feature_names)

    torch.manual_seed(42)
    model = LogisticRegressionModel(train_ds.X.shape[1], num_classes=2)
    history = train_model(model, DataLoader(train_ds, batch_size=32, shuffle=True),
                          DataLoader(val_ds, batch_size=256), nn.CrossEntropyLoss(),
                          epochs=300, lr=0.05, early_stopping=EarlyStopping(patience=20))

    metrics = classification_metrics(test_ds.y, model.predict_proba(test_ds.X))
    print("2.2 Логистическая регрессия, тест:")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {key:<10} {metrics[key]:.4f}")
    # Влияние признака на шанс выжить = вес класса 1 минус вес класса 0
    w = model.linear.weight.detach()
    effect = pd.Series((w[1] - w[0]).numpy(), index=train_ds.feature_names)
    print("Влияние на шанс выжить:")
    print(effect.sort_values(key=abs, ascending=False).round(3).to_string())
    plot_confusion_matrix(metrics["confusion_matrix"], ["погиб", "выжил"],
                          PLOTS / "2_2_confusion_matrix_titanic.png", "2.2 Confusion matrix (titanic)")
    plot_curves(history, PLOTS / "2_2_loss_titanic.png", "2.2 Loss (titanic)")
    torch.save(model.state_dict(), MODELS / "logreg_titanic.pt")


if __name__ == "__main__":
    setup_logging()
    run_regression_on_csv()
    run_classification_on_csv()
