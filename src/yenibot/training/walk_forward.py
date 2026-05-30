from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class FoldIndices:
    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class PurgedWalkForwardCV:
    def __init__(
        self,
        *,
        train_bars: int,
        val_bars: int,
        test_bars: int,
        step_bars: int,
        purge_bars: int,
        embargo_bars: int,
    ) -> None:
        self.train_bars = train_bars
        self.val_bars = val_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

    def split(self, n_rows: int) -> Iterator[FoldIndices]:
        fold = 0
        start = 0
        while True:
            train_start = start
            train_end = train_start + self.train_bars
            val_start = train_end + self.purge_bars
            val_end = val_start + self.val_bars
            test_start = val_end + self.embargo_bars
            test_end = test_start + self.test_bars
            if test_end > n_rows:
                break
            yield FoldIndices(
                fold=fold,
                train=np.arange(train_start, train_end),
                val=np.arange(val_start, val_end),
                test=np.arange(test_start, test_end),
            )
            fold += 1
            start += self.step_bars
