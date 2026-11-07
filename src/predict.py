import numpy as np
import torch

from src import config
from src.data import load_raw_data, preprocess
from src.model import Classifier


def load_model():
    model = Classifier()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(model, X, scaler):
    """Devuelve la clase predicha para nuevas muestras X (array de forma [n, 2])."""
    X_scaled = scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_scaled))
        probs = torch.softmax(logits, dim=1)
    return probs.argmax(dim=1).numpy(), probs.numpy()


if __name__ == "__main__":
    # El scaler se reconstruye con los mismos datos de entrenamiento.
    X, y = load_raw_data()
    _, _, _, _, scaler = preprocess(X, y)

    model = load_model()
    nuevas = np.array([[0.0, 0.5], [1.5, -0.5]])
    clases, probs = predict(model, nuevas, scaler)
    for x, c, p in zip(nuevas, clases, probs):
        print(f"Entrada {x} -> clase {c} (confianza {p.max():.2%})")
