from __future__ import annotations

import pandas as pd

from yenibot.labeling import add_long_only_labels


def test_long_only_triple_barrier_labels_and_drops_tail() -> None:
    timestamps = pd.date_range("2022-01-01", periods=16, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * 16,
            "high": [100.5] * 16,
            "low": [99.5] * 16,
            "close": [100.0] * 16,
            "atr_14": [1.0] * 16,
        }
    )
    frame.loc[1, "high"] = 103.0
    labeled = add_long_only_labels(frame, max_holding_bars=5)

    assert len(labeled) == 11
    assert labeled.loc[0, "label"] == 1
    assert labeled.loc[0, "hit_type"] == "tp"
    assert labeled.loc[0, "tb_return"] > 0
