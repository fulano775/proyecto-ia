import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src import config


def load_raw_data():
    """Carga datos. Por defecto genera un dataset de ejemplo (make_moons).
    Sustituye esta función por la lectura de tus archivos en data/raw/."""
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=config.SEED)
    return X.astype(np.float32), y.astype(np.int64)


def preprocess(X, y):
    """Divide en train/test y normaliza con los datos de entrenamiento."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test, scaler


def make_loader(X, y, shuffle=True):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle)
