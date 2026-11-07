import torch
import torch.nn as nn

from src import config
from src.data import load_raw_data, preprocess, make_loader
from src.model import Classifier


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


def train():
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_raw_data()
    X_train, X_test, y_train, y_test, _ = preprocess(X, y)
    train_loader = make_loader(X_train, y_train, shuffle=True)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        if epoch % 10 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            print(f"Época {epoch:3d} | pérdida {total_loss / len(X_train):.4f} | precisión test {acc:.2%}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Modelo guardado en {config.MODEL_PATH}")


if __name__ == "__main__":
    train()
