from __future__ import annotations

import zipfile
from io import BytesIO
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

KLINE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
    "ignore",
]

BINANCE_VISION_BASE_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
]

FUTURES_METRICS_COLUMNS = [
    "timestamp",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

FUNDING_RATE_COLUMNS = [
    "timestamp",
    "symbol",
    "funding_rate",
    "mark_price",
]

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def interval_to_milliseconds(interval: str) -> int:
    try:
        return INTERVAL_TO_MS[interval]
    except KeyError as exc:
        raise ValueError(f"Unsupported Binance interval: {interval}") from exc


def to_milliseconds(value: str | int | float | datetime | pd.Timestamp | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return int(ts.timestamp() * 1000)


def klines_to_dataframe(rows: list[list[Any]]) -> pd.DataFrame:
    """Convert Binance full kline rows into a typed DataFrame."""

    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    if any(len(row) < len(KLINE_COLUMNS) for row in rows):
        raise ValueError("Binance kline rows must contain the full 12-column schema")

    frame = pd.DataFrame([row[: len(KLINE_COLUMNS)] for row in rows], columns=KLINE_COLUMNS)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise")
    frame["close_time"] = pd.to_numeric(frame["close_time"], errors="raise")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["num_trades"] = frame["num_trades"].astype("int64")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _is_restricted_location_error(exc: requests.HTTPError) -> bool:
    response = exc.response
    return response is not None and response.status_code == 451


def download_full_klines(
    symbol: str,
    interval: str,
    start: str | int | float | datetime | pd.Timestamp,
    end: str | int | float | datetime | pd.Timestamp | None = None,
    *,
    base_url: str = "https://fapi.binance.com",
    vision_base_url: str = BINANCE_VISION_BASE_URL,
    data_source: str = "auto",
    limit: int = 1500,
    request_sleep_seconds: float = 0.15,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download full Binance USDT-M futures klines with microstructure columns."""

    if data_source not in {"auto", "rest", "vision"}:
        raise ValueError("data_source must be one of: auto, rest, vision")
    if data_source == "vision":
        return download_full_klines_from_vision(
            symbol,
            interval,
            start,
            end,
            vision_base_url=vision_base_url,
            session=session,
        )

    if limit > 1500:
        raise ValueError("Binance futures kline limit cannot exceed 1500")
    interval_ms = interval_to_milliseconds(interval)
    start_ms = to_milliseconds(start)
    end_ms = to_milliseconds(end)
    if start_ms is None:
        raise ValueError("start is required")
    if end_ms is not None and end_ms <= start_ms:
        raise ValueError("end must be after start")

    http = session or requests.Session()
    url = f"{base_url.rstrip('/')}/fapi/v1/klines"
    rows: list[list[Any]] = []
    cursor = start_ms

    while True:
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "limit": limit,
        }
        if end_ms is not None:
            params["endTime"] = end_ms - 1

        response = http.get(url, params=params, timeout=30)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if data_source == "auto" and _is_restricted_location_error(exc):
                return download_full_klines_from_vision(
                    symbol,
                    interval,
                    start,
                    end,
                    vision_base_url=vision_base_url,
                    session=http,
                )
            raise
        batch = response.json()
        if not isinstance(batch, list):
            raise ValueError(f"Unexpected Binance response: {batch}")
        if not batch:
            break

        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_cursor = last_open_time + interval_ms
        if next_cursor <= cursor:
            raise RuntimeError("Binance pagination did not advance")
        cursor = next_cursor
        if end_ms is not None and cursor >= end_ms:
            break
        if len(batch) < limit:
            break
        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)

    return klines_to_dataframe(rows)


def download_full_klines_from_vision(
    symbol: str,
    interval: str,
    start: str | int | float | datetime | pd.Timestamp,
    end: str | int | float | datetime | pd.Timestamp | None = None,
    *,
    vision_base_url: str = BINANCE_VISION_BASE_URL,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download full USDT-M futures klines from Binance Vision bulk zip files.

    This path is used when Colab or another host receives HTTP 451 from the
    Binance Futures REST API. Binance Vision kline CSVs contain the same
    12-column schema needed for taker/order-flow features.
    """

    start_ms = to_milliseconds(start)
    end_ms = to_milliseconds(end)
    if start_ms is None:
        raise ValueError("start is required")
    if end_ms is None:
        end_ms = int(pd.Timestamp.now(tz=timezone.utc).timestamp() * 1000)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    http = session or requests.Session()
    rows: list[list[Any]] = []
    for year, month in _month_iter(start_ms, end_ms):
        monthly_url = _vision_monthly_url(vision_base_url, symbol, interval, year, month)
        month_rows = _download_vision_zip(monthly_url, session=http)
        if month_rows is None:
            month_rows = []
            for day in _days_in_month(year, month, start_ms, end_ms):
                daily_url = _vision_daily_url(vision_base_url, symbol, interval, day)
                daily_rows = _download_vision_zip(daily_url, session=http)
                if daily_rows is not None:
                    month_rows.extend(daily_rows)
        rows.extend(month_rows)

    if not rows:
        raise ValueError(
            "No Binance Vision kline files were found for the requested range. "
            "Check symbol, interval, and date range."
        )
    filtered = [row for row in rows if start_ms <= int(row[0]) < end_ms]
    return klines_to_dataframe(filtered)


def futures_metrics_to_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert Binance Vision futures metrics CSV rows into a typed frame."""

    if frame.empty:
        return pd.DataFrame(columns=FUTURES_METRICS_COLUMNS)
    df = frame.copy()
    if "timestamp" not in df.columns:
        if "create_time" not in df.columns:
            raise ValueError("Futures metrics must contain create_time or timestamp")
        df = df.rename(columns={"create_time": "timestamp"})
    missing = [column for column in FUTURES_METRICS_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing Binance futures metrics columns: {missing}")
    df = df[FUTURES_METRICS_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for column in FUTURES_METRICS_COLUMNS:
        if column in {"timestamp", "symbol"}:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def download_futures_metrics_from_vision(
    symbol: str,
    start: str | int | float | datetime | pd.Timestamp,
    end: str | int | float | datetime | pd.Timestamp | None = None,
    *,
    vision_base_url: str = BINANCE_VISION_BASE_URL,
    request_sleep_seconds: float = 0.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download Binance Vision USDT-M futures metrics zips.

    The metrics archive contains 5-minute snapshots such as open interest and
    long/short ratios. It is the historical source we need because the REST
    open-interest statistics endpoint only exposes the latest month.
    """

    start_ms = to_milliseconds(start)
    end_ms = to_milliseconds(end)
    if start_ms is None:
        raise ValueError("start is required")
    if end_ms is None:
        end_ms = int(pd.Timestamp.now(tz=timezone.utc).timestamp() * 1000)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    http = session or requests.Session()
    frames: list[pd.DataFrame] = []
    start_day = pd.to_datetime(start_ms, unit="ms", utc=True).floor("D")
    end_day = pd.to_datetime(end_ms - 1, unit="ms", utc=True).floor("D")
    for day in pd.date_range(start_day, end_day, freq="D", tz=timezone.utc):
        url = _vision_metrics_daily_url(vision_base_url, symbol, day)
        frame = _download_vision_csv_zip(url, session=http)
        if frame is not None and not frame.empty:
            frames.append(frame)
        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)

    if not frames:
        raise ValueError(
            "No Binance Vision futures metrics files were found for the requested range. "
            "Check symbol and date range."
        )
    df = futures_metrics_to_dataframe(pd.concat(frames, ignore_index=True))
    start_ts = pd.to_datetime(start_ms, unit="ms", utc=True)
    end_ts = pd.to_datetime(end_ms, unit="ms", utc=True)
    return df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].reset_index(drop=True)


def funding_rates_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert Binance funding rate REST rows into a typed DataFrame."""

    if not rows:
        return pd.DataFrame(columns=FUNDING_RATE_COLUMNS)
    frame = pd.DataFrame(rows)
    required = ["symbol", "fundingRate", "fundingTime"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing Binance funding rate columns: {missing}")
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pd.to_numeric(frame["fundingTime"], errors="raise"), unit="ms", utc=True),
            "symbol": frame["symbol"].astype(str),
            "funding_rate": pd.to_numeric(frame["fundingRate"], errors="raise"),
            "mark_price": pd.to_numeric(frame.get("markPrice"), errors="coerce") if "markPrice" in frame else np.nan,
        }
    )
    return out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def download_funding_rates(
    symbol: str,
    start: str | int | float | datetime | pd.Timestamp,
    end: str | int | float | datetime | pd.Timestamp | None = None,
    *,
    base_url: str = "https://fapi.binance.com",
    limit: int = 1000,
    request_sleep_seconds: float = 0.15,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download Binance USDT-M funding rate history from REST."""

    if limit > 1000:
        raise ValueError("Binance funding rate limit cannot exceed 1000")
    start_ms = to_milliseconds(start)
    end_ms = to_milliseconds(end)
    if start_ms is None:
        raise ValueError("start is required")
    if end_ms is None:
        end_ms = int(pd.Timestamp.now(tz=timezone.utc).timestamp() * 1000)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    http = session or requests.Session()
    url = f"{base_url.rstrip('/')}/fapi/v1/fundingRate"
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": limit,
        }
        response = http.get(url, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise ValueError(f"Unexpected Binance funding response: {batch}")
        if not batch:
            break
        rows.extend(batch)
        last_time = int(batch[-1]["fundingTime"])
        next_cursor = last_time + 1
        if next_cursor <= cursor:
            raise RuntimeError("Binance funding pagination did not advance")
        cursor = next_cursor
        if len(batch) < limit:
            break
        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)
    return funding_rates_to_dataframe(rows)


def _vision_monthly_url(base_url: str, symbol: str, interval: str, year: int, month: int) -> str:
    return (
        f"{base_url.rstrip('/')}/data/futures/um/monthly/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    )


def _vision_daily_url(base_url: str, symbol: str, interval: str, day: pd.Timestamp) -> str:
    return (
        f"{base_url.rstrip('/')}/data/futures/um/daily/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip"
    )


def _vision_metrics_daily_url(base_url: str, symbol: str, day: pd.Timestamp) -> str:
    return (
        f"{base_url.rstrip('/')}/data/futures/um/daily/metrics/"
        f"{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"
    )


def _download_vision_zip(url: str, *, session: requests.Session) -> list[list[Any]] | None:
    response = session.get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV inside Binance Vision zip: {url}")
        with archive.open(csv_names[0]) as handle:
            raw = pd.read_csv(handle, header=None)
    if raw.empty:
        return []
    raw = raw.iloc[:, : len(KLINE_COLUMNS)].copy()
    raw.columns = KLINE_COLUMNS
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    raw["timestamp"] = raw["timestamp"].astype("int64")
    return raw.values.tolist()


def _download_vision_csv_zip(url: str, *, session: requests.Session) -> pd.DataFrame | None:
    response = session.get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV inside Binance Vision zip: {url}")
        with archive.open(csv_names[0]) as handle:
            return pd.read_csv(handle)


def _month_iter(start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    start_ts = pd.to_datetime(start_ms, unit="ms", utc=True)
    end_ts = pd.to_datetime(end_ms - 1, unit="ms", utc=True)
    start_period = pd.Period(year=start_ts.year, month=start_ts.month, freq="M")
    end_period = pd.Period(year=end_ts.year, month=end_ts.month, freq="M")
    months = pd.period_range(start_period, end_period, freq="M")
    return [(period.year, period.month) for period in months]


def _days_in_month(year: int, month: int, start_ms: int, end_ms: int) -> list[pd.Timestamp]:
    month_start = pd.Timestamp(year=year, month=month, day=1, tz=timezone.utc)
    month_end = month_start + pd.offsets.MonthBegin(1)
    start_ts = max(pd.to_datetime(start_ms, unit="ms", utc=True).floor("D"), month_start)
    end_ts = min(pd.to_datetime(end_ms - 1, unit="ms", utc=True).floor("D"), month_end - pd.Timedelta(days=1))
    if end_ts < start_ts:
        return []
    return list(pd.date_range(start_ts, end_ts, freq="D", tz=timezone.utc))
