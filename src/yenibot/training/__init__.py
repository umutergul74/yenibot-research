"""Training utilities."""

from yenibot.training.dataset import SequenceDataset
from yenibot.training.walk_forward import FoldIndices, PurgedWalkForwardCV

__all__ = [
    "SequenceDataset",
    "FoldIndices",
    "PurgedWalkForwardCV",
    "run_walk_forward_training",
    "set_random_seed",
    "train_one_fold",
]


def __getattr__(name: str):
    if name in {"run_walk_forward_training", "set_random_seed", "train_one_fold"}:
        import importlib

        module = importlib.import_module("yenibot.training.trainer")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
