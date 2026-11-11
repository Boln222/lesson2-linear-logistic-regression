"""Задание 1.
1.1 Линейная регрессия + L1/L2-регуляризация + early stopping (датасет diabetes).
1.2 Логистическая регрессия на несколько классов + precision, recall, F1, ROC-AUC
    и confusion matrix (датасет wine, 3 класса).
"""
import copy
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # графики сохраняем в файл, без открытия окон
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

PLOTS = Path(__file__).parent / "plots"
MODELS = Path(__file__).parent / "models"
OPTIMIZERS = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}
logger = logging.getLogger(__name__)


def setup_logging():
    """Логи вида 'время | уровень | сообщение' + папки для графиков и моделей."""
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s | %(levelname)s | %(message)s")
    PLOTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)


# ================= 1.1 Линейная регрессия: L1, L2, early stopping =================

class LinearRegressionModel(nn.Module):    #признаки * веса -> number 
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1) 


def regularization_penalty(model, l1=0.0, l2=0.0):  # штраф за большие веса 
    w = model.linear.weight
    return l1 * w.abs().sum() + l2 * (w ** 2).sum()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta  
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state = None       
        self.bad_epochs = 0         

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore_best(self, model):
        model.load_state_dict(self.best_state)


def evaluate_loss(model, loader, loss_fn):
    model.eval()
    with torch.no_grad():
        total = sum(loss_fn(model(X), y).item() * len(X) for X, y in loader)
    return total / len(loader.dataset)


def train_model(model, train_loader, val_loader, loss_fn, epochs=200, lr=0.05,
                optimizer="sgd", l1=0.0, l2=0.0, early_stopping=None, log_every=20):
    opt = OPTIMIZERS[optimizer](model.parameters(), lr=lr)
    history = {"train": [], "val": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            opt.zero_grad()                                
            data_loss = loss_fn(model(X), y)               
            loss = data_loss + regularization_penalty(model, l1, l2)
            loss.backward()                                
            opt.step()                                     
            total += data_loss.item() * len(X)            
        history["train"].append(total / len(train_loader.dataset))
        history["val"].append(evaluate_loss(model, val_loader, loss_fn))

        if log_every and epoch % log_every == 0:
            logger.info("Эпоха %d: train=%.4f, val=%.4f", epoch, history["train"][-1], history["val"][-1])
        if early_stopping and early_stopping.step(history["val"][-1], model, epoch):
            logger.info("Early stopping на эпохе %d, лучшая эпоха %d", epoch, early_stopping.best_epoch)
            break
    if early_stopping:
        early_stopping.restore_best(model)
    return history


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X)


def regression_metrics(y_true, y_pred):    #насколько хорошо угадано число 
    mse = ((y_pred - y_true) ** 2).mean()
    return {"rmse": mse.sqrt().item(),
            "mae": (y_pred - y_true).abs().mean().item(),
            "r2": (1 - mse / ((y_true - y_true.mean()) ** 2).mean()).item()}


def split_tensors(X, y, stratify=False, seed=42):   #разделить данные на train / val / test
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if stratify else None)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25, random_state=seed, stratify=y_rest if stratify else None)
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize(train, *others):
    mean, std = train.mean(dim=0), train.std(dim=0)
    return [(t - mean) / std for t in (train, *others)] + [mean, std]


# ================= 1.2 Логистическая регрессия на несколько классов =================

class LogisticRegressionModel(nn.Module):      #ставим баллы каждому классу и переводим в проценты (softmax) - wine
    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x) 

    def predict_proba(self, x):
        return torch.softmax(predict(self, x), dim=1)


def confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def precision_recall_f1(cm):   #precision - делим на столбец, recall - на строку 
    cm = cm.float()
    tp = cm.diag()                                 # верно найденные
    precision = tp / cm.sum(dim=0).clamp(min=1)   
    recall = tp / cm.sum(dim=1).clamp(min=1)   
    f1 = torch.nan_to_num(2 * precision * recall / (precision + recall))  # 0/0 -> 0
    return precision, recall, f1


def roc_auc_binary(y_true, scores):
    diff = scores[y_true == 1].unsqueeze(1) - scores[y_true == 0].unsqueeze(0)  # все пары
    return ((diff > 0).float() + 0.5 * (diff == 0).float()).mean().item()


def roc_auc(y_true, probs):
    n = probs.shape[1]
    if n == 2:
        return roc_auc_binary(y_true, probs[:, 1])
    return sum(roc_auc_binary((y_true == k).long(), probs[:, k]) for k in range(n)) / n


def classification_metrics(y_true, probs):
    y_pred = probs.argmax(dim=1)
    cm = confusion_matrix(y_true, y_pred, probs.shape[1])
    precision, recall, f1 = precision_recall_f1(cm)
    k = 1 if probs.shape[1] == 2 else slice(None)  # класс 1 или все классы
    return {"accuracy": (y_pred == y_true).float().mean().item(),
            "precision": precision[k].mean().item(),
            "recall": recall[k].mean().item(),
            "f1": f1[k].mean().item(),
            "roc_auc": roc_auc(y_true, probs),
            "confusion_matrix": cm}


# ================= Графики =================

def plot_curves(curves, path, title, xlabel="Эпоха", ylabel="Loss", log=False, ylim=None):
    plt.figure(figsize=(8, 5))
    for label, values in curves.items():
        plt.plot(range(1, len(values) + 1), values, label=label)
    if log:
        plt.yscale("log")
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, class_names, path, title):
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Настоящий класс")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j].item(), ha="center", va="center", color="red")
    plt.title(title)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ================= Запуск =================

def expand_polynomial(X):
    n = X.shape[1]
    products = [X[:, i] * X[:, j] for i in range(n) for j in range(i, n)]
    return torch.cat([X, torch.stack(products, dim=1)], dim=1)


def run_linear_regression_demo():
    data = load_diabetes()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32)
    X_train, X_val, X_test, y_train, y_val, y_test = split_tensors(X, y)
    X_train, X_val, X_test, _, _ = standardize(X_train, X_val, X_test)
    X_train, X_val, X_test, _, _ = standardize(*(expand_polynomial(t) for t in (X_train, X_val, X_test)))
    y_train, y_val, y_test, y_mean, y_std = standardize(y_train, y_val, y_test)  # y тоже, чтобы loss ~1
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)

    configs = {  # название: (l1, l2, early stopping?)
        "без регуляризации": (0.0, 0.0, False),
        "early stopping": (0.0, 0.0, True),
        "L1": (0.02, 0.0, False),
        "L2": (0.0, 0.05, False),
        "L1 + L2 + early stopping": (0.01, 0.02, True),
    }
    rows, val_curves, weights = [], {}, {}
    for name, (l1, l2, use_es) in configs.items():
        logger.info("--- %s ---", name)
        torch.manual_seed(42)
        model = LinearRegressionModel(X_train.shape[1])
        stopper = EarlyStopping(patience=30) if use_es else None
        history = train_model(model, train_loader, val_loader, nn.MSELoss(), epochs=600, lr=0.01,
                              l1=l1, l2=l2, early_stopping=stopper, log_every=100)
        metrics = regression_metrics(y_test * y_std + y_mean, predict(model, X_test) * y_std + y_mean)
        w = model.linear.weight.detach().squeeze(0)
        rows.append({"модель": name, "эпох": len(history["val"]),
                     "лучшая эпоха": stopper.best_epoch if stopper else "-",
                     "test RMSE": metrics["rmse"], "test R2": metrics["r2"],
                     "сумма |w|": w.abs().sum().item(), "|w| < 0.01": int((w.abs() < 0.01).sum())})
        val_curves[name] = history["val"]
        weights[name] = w.abs().sort(descending=True).values.tolist()
        if name == "без регуляризации":
            plot_curves(history, PLOTS / "1_1_overfitting.png", "1.1 Без регуляризации: переобучение")

    print("\n1.1 Сравнение на тесте (diabetes, 65 признаков):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    plot_curves(val_curves, PLOTS / "1_1_val_loss.png", "1.1 Loss на валидации")
    plot_curves(weights, PLOTS / "1_1_weights.png", "1.1 Модули весов по убыванию",
                xlabel="Номер веса", ylabel="|w|")
    torch.save(model.state_dict(), MODELS / "linreg_diabetes.pt")


def run_logistic_regression_demo():
    """1.2: логистическая регрессия на 3 класса и все метрики."""
    data = load_wine()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.long)  # номера классов 0, 1, 2
    X_train, X_val, X_test, y_train, y_val, y_test = split_tensors(X, y, stratify=True)
    X_train, X_val, X_test, _, _ = standardize(X_train, X_val, X_test)

    torch.manual_seed(42)
    model = LogisticRegressionModel(X.shape[1], num_classes=3)
    history = train_model(model, DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True),
                          DataLoader(TensorDataset(X_val, y_val), batch_size=256),
                          nn.CrossEntropyLoss(), epochs=300, lr=0.05, early_stopping=EarlyStopping(patience=20))

    metrics = classification_metrics(y_test, model.predict_proba(X_test))
    print("\n1.2 Метрики на тесте (wine, 3 класса, macro):")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {key:<10} {metrics[key]:.4f}")
    print("Confusion matrix:\n", metrics["confusion_matrix"])
    plot_confusion_matrix(metrics["confusion_matrix"], ["сорт 0", "сорт 1", "сорт 2"],
                          PLOTS / "1_2_confusion_matrix_wine.png", "1.2 Confusion matrix (wine)")
    plot_curves(history, PLOTS / "1_2_loss_wine.png", "1.2 Loss (wine)")

    # Сохраняем веса (state_dict) и проверяем, что загруженная модель отвечает так же
    torch.save(model.state_dict(), MODELS / "logreg_wine.pt")
    loaded = LogisticRegressionModel(X.shape[1], num_classes=3)
    loaded.load_state_dict(torch.load(MODELS / "logreg_wine.pt"))
    assert torch.equal(loaded.predict_proba(X_test), model.predict_proba(X_test))


if __name__ == "__main__":
    setup_logging()
    run_linear_regression_demo()
    run_logistic_regression_demo()
