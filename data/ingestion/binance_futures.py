"""Binance USD-M Futures public data fetcher."""

import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import httpx
from loguru import logger

BASE_URL = "https://fapi.binance.com"
DB_PATH = Path("data/market.duckdb")


def fetch_klines(symbol="BTCUSDT", interval="1m", start_time=None, end_time=None, limit=1500):
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    with httpx.Client(timeout=30.0) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


def save_klines_to_duckdb(klines, symbol, interval, db_path=DB_PATH):
    if not klines:
        return 0
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol VARCHAR, interval VARCHAR, open_time TIMESTAMP,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, close_time TIMESTAMP, quote_volume DOUBLE,
            trades INTEGER, taker_buy_volume DOUBLE,
            taker_buy_quote_volume DOUBLE,
            PRIMARY KEY (symbol, interval, open_time)
        )
    """)
    rows = []
    for k in klines:
        rows.append((
            symbol, interval,
            datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            float(k[1]), float(k[2]), float(k[3]), float(k[4]),
            float(k[5]),
            datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc),
            float(k[7]), int(k[8]), float(k[9]), float(k[10]),
        ))
    conn.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    conn.close()
    return count


def fetch_funding_rate(symbol="BTCUSDT", start_time=None, end_time=None, limit=1000):
    url = f"{BASE_URL}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    with httpx.Client(timeout=30.0) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


def save_funding_to_duckdb(funding_data, symbol, db_path=DB_PATH):
    if not funding_data:
        return 0
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding_rate (
            symbol VARCHAR, funding_time TIMESTAMP,
            funding_rate DOUBLE, mark_price DOUBLE,
            PRIMARY KEY (symbol, funding_time)
        )
    """)
    rows = []
    for f in funding_data:
        rows.append((
            symbol,
            datetime.fromtimestamp(f["fundingTime"] / 1000, tz=timezone.utc),
            float(f["fundingRate"]),
            float(f.get("markPrice", 0.0)),
        ))
    conn.executemany("INSERT OR REPLACE INTO funding_rate VALUES (?,?,?,?)", rows)
    count = conn.execute("SELECT COUNT(*) FROM funding_rate").fetchone()[0]
    conn.close()
    return count


def fetch_open_interest_hist(symbol="BTCUSDT", period="5m", limit=500):
    url = f"{BASE_URL}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}
    with httpx.Client(timeout=30.0) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


def save_oi_to_duckdb(oi_data, symbol, period, db_path=DB_PATH):
    if not oi_data:
        return 0
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS open_interest (
            symbol VARCHAR, period VARCHAR, timestamp TIMESTAMP,
            open_interest DOUBLE, open_interest_value DOUBLE,
            PRIMARY KEY (symbol, period, timestamp)
        )
    """)
    rows = []
    for oi in oi_data:
        rows.append((
            symbol, period,
            datetime.fromtimestamp(oi["timestamp"] / 1000, tz=timezone.utc),
            float(oi["sumOpenInterest"]),
            float(oi["sumOpenInterestValue"]),
        ))
    conn.executemany("INSERT OR REPLACE INTO open_interest VALUES (?,?,?,?,?)", rows)
    count = conn.execute("SELECT COUNT(*) FROM open_interest").fetchone()[0]
    conn.close()
    return count


def main_full():
    symbol = "BTCUSDT"
    end_time = int(time.time() * 1000)

    logger.info("Step 1/3: Fetching OHLCV...")
    start_time = end_time - (24 * 60 * 60 * 1000)
    all_klines = []
    current_start = start_time
    while current_start < end_time:
        klines = fetch_klines(symbol, "1m", current_start, end_time, 1500)
        if not klines:
            break
        all_klines.extend(klines)
        current_start = klines[-1][6] + 1
        time.sleep(0.2)
    c1 = save_klines_to_duckdb(all_klines, symbol, "1m")
    logger.success(f"OHLCV rows: {c1}")

    logger.info("Step 2/3: Fetching funding rate...")
    fs = end_time - (7 * 24 * 60 * 60 * 1000)
    funding = fetch_funding_rate(symbol, fs, end_time, 1000)
    c2 = save_funding_to_duckdb(funding, symbol)
    logger.success(f"Funding rows: {c2}")

    logger.info("Step 3/3: Fetching open interest...")
    oi_data = fetch_open_interest_hist(symbol, "5m", limit=500)
    c3 = save_oi_to_duckdb(oi_data, symbol, "5m")
    logger.success(f"OI rows: {c3}")

    logger.success("All data fetched successfully!")


if __name__ == "__main__":
    main_full()
