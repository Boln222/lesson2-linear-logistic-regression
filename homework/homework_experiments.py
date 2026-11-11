"""Задание 3 (линейная регрессия на data/insurance.csv).
3.1 Гиперпараметры: learning rate, размер батча, оптимизаторы SGD / Adam / RMSprop.
3.2 Feature engineering: полиномиальные признаки, взаимодействия, статистики по группам.
"""
import math
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from homework_datasets import DATA, CSVDataset, make_datasets, regression_metrics_in_dollars, split_dataframe
from homework_model_modification import (PLOTS, EarlyStopping, LinearRegressionModel, plot_curves,
                                         setup_logging, train_model)


# ================= 3.1 Гиперпараметры =================

def train_once(train_ds, val_ds, optimizer, lr, batch_size, epochs=100):
    """Обучает модель ровно 100 эпох (без early stopping, чтобы кривые были одной длины)."""
    torch.manual_seed(42)  # одинаковые начальные веса у всех запусков
    model = LinearRegressionModel(train_ds.X.shape[1])
    start = time.time()
    history = train_model(model, DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                          DataLoader(val_ds, batch_size=512), nn.MSELoss(),
                          epochs=epochs, lr=lr, optimizer=optimizer, log_every=0)
    return history["val"], time.time() - start


def run_hyperparameter_experiments():
    """Меняем по одному параметру, остальные фиксированы. Итог — таблица и графики."""
    train_ds, val_ds, _ = make_datasets(pd.read_csv(DATA / "insurance.csv"), "charges")
    experiments = {  # имя: список (оптимизатор, lr, batch)
        "learning_rate": [("sgd", lr, 32) for lr in (0.0001, 0.001, 0.01, 0.1, 1.0)],
        "batch_size": [("sgd", 0.01, bs) for bs in (8, 32, 128, len(train_ds))],
        "optimizers": [(opt, lr, 32) for lr in (0.001, 0.01, 0.1) for opt in ("sgd", "adam", "rmsprop")],
    }
    rows = []
    for name, configs in experiments.items():
        curves = {}
        for optimizer, lr, batch_size in configs:
            val_loss, seconds = train_once(train_ds, val_ds, optimizer, lr, batch_size)
            reached = [epoch for epoch, loss in enumerate(val_loss, 1) if loss <= 0.35]
            rows.append({"эксперимент": name, "оптимизатор": optimizer, "lr": lr, "batch": batch_size,
                         "val MSE": round(val_loss[-1], 4) if math.isfinite(val_loss[-1]) else "разошлось",
                         "эпох до 0.35": reached[0] if reached else "не достиг",
                         "время, с": round(seconds, 2)})
            if name != "optimizers" or lr == 0.001:  # оптимизаторы рисуем при lr=0.001
                curves[f"{optimizer}, lr={lr}, batch={batch_size}"] = val_loss
        plot_curves(curves, PLOTS / f"3_1_{name}.png", f"3.1 {name}: loss на валидации",
                    log=True, ylim=(0.3, 3))

    table = pd.DataFrame(rows)
    print("\n3.1 Гиперпараметры (val MSE на стандартизованном ответе; 1.0 = 'всегда среднее'):")
    print(table.to_string(index=False))
    table.to_csv(PLOTS / "3_1_results.csv", index=False)


# ================= 3.2 Feature engineering =================

def as_number(column):
    """Числа оставляем, бинарный текст (no/yes) превращаем в 0/1."""
    if pd.api.types.is_numeric_dtype(column):
        return column.astype(float)
    return (column == sorted(column.unique())[-1]).astype(float)


def add_polynomial_features(df, columns):
    """Полиномиальные: x² — линейная модель сможет учесть изгиб зависимости."""
    df = df.copy()
    for col in columns:
        df[f"{col}^2"] = as_number(df[col]) ** 2
    return df


def add_interaction_features(df, pairs):
    """Взаимодействия: a * b — например, ИМТ важен именно у курильщиков (bmi * smoker)."""
    df = df.copy()
    for a, b in pairs:
        df[f"{a}*{b}"] = as_number(df[a]) * as_number(df[b])
    return df


def add_group_statistics(train, others, group, col):
    """Статистические: среднее и дисперсия col в группе человека + его отклонение от среднего.

    Считаются только по train и переносятся на val/test по названию группы.
    """
    mean = train.groupby(group)[col].mean()
    var = train.groupby(group)[col].var()
    result = []
    for df in [train, *others]:
        df = df.copy()
        df[f"{col}_mean_by_{group}"] = df[group].map(mean)
        df[f"{col}_var_by_{group}"] = df[group].map(var)
        df[f"{col}_minus_mean"] = df[col] - df[f"{col}_mean_by_{group}"]
        result.append(df)
    return result


def run_feature_engineering():
    """Сравнивает базовую модель с моделями на новых признаках (одно разбиение на всех)."""
    base_parts = split_dataframe(pd.read_csv(DATA / "insurance.csv"))
    variants = {  # имя: (полиномиальные?, взаимодействия?, статистики?)
        "базовые": (False, False, False),
        "+ полиномиальные": (True, False, False),
        "+ взаимодействия": (False, True, False),
        "+ статистические": (False, False, True),
        "все вместе": (True, True, True),
    }
    rows = []
    for name, (poly, inter, stats) in variants.items():
        parts = base_parts
        if poly:
            parts = [add_polynomial_features(df, ["age", "bmi", "children"]) for df in parts]
        if inter:
            parts = [add_interaction_features(df, [("age", "bmi"), ("bmi", "smoker"), ("age", "smoker")])
                     for df in parts]
        if stats:
            parts = add_group_statistics(parts[0], parts[1:], "region", "bmi")

        train_ds = CSVDataset(parts[0], "charges")
        val_ds = CSVDataset(parts[1], "charges", reference=train_ds)
        test_ds = CSVDataset(parts[2], "charges", reference=train_ds)
        torch.manual_seed(42)
        model = LinearRegressionModel(train_ds.X.shape[1])
        train_model(model, DataLoader(train_ds, batch_size=32, shuffle=True), DataLoader(val_ds, batch_size=512),
                    nn.MSELoss(), epochs=300, lr=0.01, early_stopping=EarlyStopping(patience=20), log_every=0)
        metrics = regression_metrics_in_dollars(model, test_ds)
        rows.append({"признаки": name, "число признаков": train_ds.X.shape[1],
                     "test R2": round(metrics["r2"], 4), "test MAE, $": round(metrics["mae"])})

    table = pd.DataFrame(rows)
    table["прирост R2"] = (table["test R2"] - table["test R2"][0]).round(4)
    print("\n3.2 Feature engineering:")
    print(table.to_string(index=False))
    table.to_csv(PLOTS / "3_2_results.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.bar(table["признаки"], table["test R2"])
    plt.ylim(0.75, 0.88)
    plt.xticks(rotation=15)
    plt.ylabel("R² на тесте")
    plt.title("3.2 Качество с новыми признаками")
    plt.savefig(PLOTS / "3_2_feature_engineering.png", dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    setup_logging()
    run_hyperparameter_experiments()
    run_feature_engineering()
