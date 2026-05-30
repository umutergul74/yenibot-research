from __future__ import annotations

import zipfile
from io import BytesIO

import pyarrow.parquet as pq
import pandas as pd
import pytest
import requests

from yenibot.data import (
    download_full_klines,
    download_futures_metrics_from_vision,
    funding_rates_to_dataframe,
    futures_metrics_to_dataframe,
)
from yenibot.data.validation import validate_full_kline_frame


def test_full_kline_validation_rejects_ohlcv_only() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC"),
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10, 11, 12],
        }
    )
    with pytest.raises(ValueError, match="Missing Binance full-kline columns"):
        validate_full_kline_frame(frame, "1h")


def test_full_kline_validation_accepts_microstructure_columns(synthetic_klines) -> None:
    frame = synthetic_klines(12, "1h")
    validated = validate_full_kline_frame(frame, "1h")
    assert len(validated) == 12
    assert validated["taker_buy_base_vol"].sum() > 0


def test_full_kline_validation_can_drop_zero_volume_rows(synthetic_klines) -> None:
    frame = synthetic_klines(6, "1h")
    frame.loc[2, ["volume", "quote_volume", "num_trades", "taker_buy_base_vol", "taker_buy_quote_vol"]] = 0

    with pytest.raises(ValueError, match="Zero or negative volume/trade activity"):
        validate_full_kline_frame(frame, "1h")

    validated = validate_full_kline_frame(frame, "1h", zero_volume_policy="drop")

    assert len(validated) == 5
    assert validated.attrs["dropped_zero_volume_rows"] == 1
    assert pd.Timestamp("2022-01-01 02:00", tz="UTC") not in set(validated["timestamp"])


def test_intrabar_validation_can_use_wider_gap_tolerance(synthetic_klines) -> None:
    frame = synthetic_klines(12, "15m")
    frame = frame.drop(index=[3, 4]).reset_index(drop=True)

    with pytest.raises(ValueError, match="Kline gap exceeds allowed threshold"):
        validate_full_kline_frame(frame, "15m", max_gap_multiplier=2)

    validated = validate_full_kline_frame(frame, "15m", max_gap_multiplier=8)

    assert len(validated) == 10
    assert validated.attrs["gap_count_gt_expected"] == 1
    assert validated.attrs["max_gap"] == "0 days 00:45:00"
    assert validated.attrs["max_gap_seconds"] == 2700.0


def test_validation_attrs_are_parquet_serializable(synthetic_klines, tmp_path) -> None:
    frame = synthetic_klines(12, "15m").drop(index=[3, 4]).reset_index(drop=True)
    validated = validate_full_kline_frame(frame, "15m", max_gap_multiplier=8)
    path = tmp_path / "klines.parquet"

    validated.to_parquet(path, index=False)
    loaded = pq.read_table(path).to_pandas()

    assert len(loaded) == len(validated)


def test_download_full_klines_falls_back_to_vision_on_451() -> None:
    session = _FakeSession()
    df = download_full_klines(
        "BTCUSDT",
        "1h",
        "2022-01-01",
        "2022-01-01 03:00",
        data_source="auto",
        session=session,
    )

    assert len(df) == 3
    assert df["taker_buy_base_vol"].sum() > 0
    assert any("data.binance.vision" in call for call in session.calls)


def test_futures_metrics_parser_requires_binance_vision_schema() -> None:
    frame = pd.DataFrame(
        {
            "create_time": ["2022-01-01 00:00:00"],
            "symbol": ["BTCUSDT"],
            "sum_open_interest": ["100.0"],
            "sum_open_interest_value": ["5000000.0"],
            "count_toptrader_long_short_ratio": ["1.2"],
            "sum_toptrader_long_short_ratio": ["0.9"],
            "count_long_short_ratio": ["1.1"],
            "sum_taker_long_short_vol_ratio": ["1.05"],
        }
    )

    parsed = futures_metrics_to_dataframe(frame)

    assert list(parsed.columns) == [
        "timestamp",
        "symbol",
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    assert parsed.loc[0, "timestamp"] == pd.Timestamp("2022-01-01 00:00:00", tz="UTC")
    assert parsed.loc[0, "sum_open_interest"] == pytest.approx(100.0)

    with pytest.raises(ValueError, match="Missing Binance futures metrics columns"):
        futures_metrics_to_dataframe(frame.drop(columns=["sum_taker_long_short_vol_ratio"]))


def test_download_futures_metrics_from_vision_reads_daily_zip() -> None:
    session = _MetricsFakeSession()

    parsed = download_futures_metrics_from_vision(
        "BTCUSDT",
        "2022-01-01",
        "2022-01-01 00:15",
        session=session,
    )

    assert len(parsed) == 3
    assert parsed["sum_open_interest"].iloc[-1] == pytest.approx(102.0)
    assert any("/daily/metrics/BTCUSDT/" in call for call in session.calls)


def test_funding_rates_parser_converts_rest_rows() -> None:
    parsed = funding_rates_to_dataframe(
        [
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.0001",
                "fundingTime": 1640995200000,
                "markPrice": "47000.0",
            }
        ]
    )

    assert list(parsed.columns) == ["timestamp", "symbol", "funding_rate", "mark_price"]
    assert parsed.loc[0, "timestamp"] == pd.Timestamp("2022-01-01 00:00:00", tz="UTC")
    assert parsed.loc[0, "funding_rate"] == pytest.approx(0.0001)


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", payload: object | None = None) -> None:
        self.status_code = status_code
        self.content = content
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self
            raise error

    def json(self) -> object:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str, **kwargs) -> _FakeResponse:
        self.calls.append(url)
        if "fapi/v1/klines" in url:
            return _FakeResponse(451)
        if "monthly/klines" in url:
            return _FakeResponse(200, _vision_zip_bytes())
        return _FakeResponse(404)


class _MetricsFakeSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str, **kwargs) -> _FakeResponse:
        self.calls.append(url)
        if "/daily/metrics/" in url:
            return _FakeResponse(200, _metrics_zip_bytes())
        return _FakeResponse(404)


def _vision_zip_bytes() -> bytes:
    rows = []
    for i in range(3):
        open_time = 1640995200000 + i * 3_600_000
        close_time = open_time + 3_600_000 - 1
        rows.append(
            [
                open_time,
                "100.0",
                "101.0",
                "99.0",
                "100.5",
                "10.0",
                close_time,
                "1005.0",
                20,
                "5.5",
                "552.75",
                "0",
            ]
        )
    csv = "\n".join(",".join(map(str, row)) for row in rows)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-1h-2022-01.csv", csv)
    return buffer.getvalue()


def _metrics_zip_bytes() -> bytes:
    rows = [
        [
            "2022-01-01 00:00:00",
            "BTCUSDT",
            "100.0",
            "5000000.0",
            "1.2",
            "0.9",
            "1.1",
            "1.05",
        ],
        [
            "2022-01-01 00:05:00",
            "BTCUSDT",
            "101.0",
            "5050000.0",
            "1.1",
            "1.0",
            "1.2",
            "0.95",
        ],
        [
            "2022-01-01 00:10:00",
            "BTCUSDT",
            "102.0",
            "5100000.0",
            "1.3",
            "1.1",
            "1.0",
            "1.15",
        ],
    ]
    columns = [
        "create_time",
        "symbol",
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    csv = ",".join(columns) + "\n" + "\n".join(",".join(map(str, row)) for row in rows)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-metrics-2022-01-01.csv", csv)
    return buffer.getvalue()
