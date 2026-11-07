import torch
import torch.nn as nn

from src import config


class Classifier(nn.Module):
    """Red neuronal feed-forward para clasificación."""

    def __init__(self, input_dim=config.INPUT_DIM,
                 hidden_dim=config.HIDDEN_DIM,
                 num_classes=config.NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
