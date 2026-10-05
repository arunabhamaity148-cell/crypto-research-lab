"""Forward return engine and first statistical test."""

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

DB_PATH = Path("data/market.duckdb")

FORWARD_HORIZONS = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "24h": 1440,
}


def add_forward_returns(df):
    df = df.copy()
    for name, bars in FORWARD_HORIZONS.items():
        df[f"fwd_ret_{name}"] = df["close"].shift(-bars) / df["close"] - 1
    return df


def run_volume_test(df, horizon="5m"):
    fwd_col = f"fwd_ret_{horizon}"
    df = df.dropna(subset=["volume_ratio", fwd_col]).copy()
    low_thresh = df["volume_ratio"].quantile(0.33)
    high_thresh = df["volume_ratio"].quantile(0.67)
    low_grp = df["volume_ratio"].le(low_thresh)
    high_grp = df["volume_ratio"].ge(high_thresh)
    low_ret = df.loc[low_grp, fwd_col]
    high_ret = df.loc[high_grp, fwd_col]
    u_stat, p_value = stats.mannwhitneyu(high_ret, low_ret, alternative="two-sided")
    return {
        "horizon": horizon,
        "n_low": len(low_ret),
        "n_high": len(high_ret),
        "mean_low": low_ret.mean(),
        "mean_high": high_ret.mean(),
        "diff_mean": high_ret.mean() - low_ret.mean(),
        "p_value": p_value,
    }


def main():
    logger.info("Loading features...")
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute("SELECT * FROM features_price ORDER BY open_time").fetchdf()
    conn.close()
    logger.info(f"Loaded {len(df)} rows")

    df = add_forward_returns(df)

    logger.info("=" * 60)
    logger.info("RESEARCH QUESTION: Does HIGH volume predict forward returns?")
    logger.info("=" * 60)

    for horizon in ["1m", "5m", "15m", "1h", "4h"]:
        r = run_volume_test(df, horizon)
        logger.info(f"Horizon {horizon}: n_low={r['n_low']} n_high={r['n_high']} diff={r['diff_mean']*100:.5f}% p={r['p_value']:.4f}")


if __name__ == "__main__":
    main()
