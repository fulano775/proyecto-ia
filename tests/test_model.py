import torch

from src import config
from src.model import Classifier


def test_forward_shape():
    model = Classifier()
    x = torch.randn(8, config.INPUT_DIM)
    out = model(x)
    assert out.shape == (8, config.NUM_CLASSES)


def test_model_has_parameters():
    model = Classifier()
    assert sum(p.numel() for p in model.parameters()) > 0
