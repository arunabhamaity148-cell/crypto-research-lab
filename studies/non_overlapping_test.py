"""Test volume signal on non-overlapping 15-minute data."""
import sys
sys.path.insert(0, ".")
import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

DB_PATH = "data/market.duckdb"


def load_and_resample():
    conn = duckdb.connect(DB_PATH, read_only=True)
    df = conn.execute("SELECT * FROM features_price ORDER BY open_time").fetchdf()
    conn.close()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.set_index("open_time")
    resampled = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "taker_buy_volume": "sum",
    }).dropna()
    return resampled


def add_features_15m(df):
    df = df.copy()
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]
    df["fwd_ret"] = df["close"].shift(-1) / df["close"] - 1
    return df


def main():
    df = load_and_resample()
    logger.info(f"Non-overlapping 15m bars: {len(df)}")
    df = add_features_15m(df).dropna()

    low = df[df["volume_ratio"] <= df["volume_ratio"].quantile(0.33)]["fwd_ret"]
    high = df[df["volume_ratio"] >= df["volume_ratio"].quantile(0.67)]["fwd_ret"]

    u, p = stats.mannwhitneyu(high, low, alternative="two-sided")
    logger.info(f"n_low={len(low)} n_high={len(high)}")
    logger.info(f"mean_low={low.mean()*100:.5f}% mean_high={high.mean()*100:.5f}%")
    logger.info(f"diff={ (high.mean()-low.mean())*100:.5f}% p={p:.4f}")
    logger.info("VERDICT: " + ("SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"))


if __name__ == "__main__":
    main()
