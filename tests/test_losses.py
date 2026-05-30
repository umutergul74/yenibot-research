from __future__ import annotations

import torch

from yenibot.losses import PairwiseLabelMarginLoss


def test_pairwise_label_margin_loss_penalizes_reversed_score_order() -> None:
    loss = PairwiseLabelMarginLoss(margin=0.25)
    good_logits = torch.tensor([2.0, 1.0, -1.0, -2.0])
    bad_logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    assert loss(good_logits, labels) < loss(bad_logits, labels)


def test_pairwise_label_margin_loss_is_zero_for_single_class_batch() -> None:
    loss = PairwiseLabelMarginLoss(margin=0.25)

    assert float(loss(torch.tensor([0.1, 0.2]), torch.tensor([1.0, 1.0]))) == 0.0
