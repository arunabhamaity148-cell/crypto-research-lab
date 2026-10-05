"""Price-based feature engineering for crypto research."""

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from loguru import logger

DB_PATH = Path("data/market.duckdb")


def load_ohlcv(symbol="BTCUSDT", interval="1m", db_path=DB_PATH) -> pd.DataFrame:
    conn = duckdb.connect(str(db_path), read_only=True)
    df = conn.execute("""
        SELECT open_time, open, high, low, close, volume,
               quote_volume, trades, taker_buy_volume
        FROM ohlcv
        WHERE symbol = ? AND interval = ?
        ORDER BY open_time
    """, [symbol, interval]).fetchdf()
    conn.close()
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=window).mean()
    return df


def add_realized_volatility(df: pd.DataFrame, window: int = 20, annualize: bool = True) -> pd.DataFrame:
    df = df.copy()
    vol = df["log_return"].rolling(window=window).std()
    if annualize:
        # 1-minute bars: 1440 bars per day, 365 days per year
        vol = vol * np.sqrt(1440 * 365)
    df["realized_vol"] = vol
    return df


def add_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["volume_ma"] = df["volume"].rolling(window=window).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]
    df["taker_buy_ratio"] = df["taker_buy_volume"] / df["volume"].replace(0, np.nan)
    return df


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature calculators in sequence."""
    df = add_returns(df)
    df = add_atr(df)
    df = add_realized_volatility(df)
    df = add_volume_features(df)
    return df


def main():
    logger.info("Loading OHLCV data...")
    df = load_ohlcv("BTCUSDT", "1m")
    logger.info(f"Loaded {len(df)} rows")

    logger.info("Building features...")
    features_df = build_price_features(df)

    logger.info("Feature columns:")
    feature_cols = ["return", "log_return", "atr", "realized_vol",
                    "volume_ratio", "taker_buy_ratio"]
    for col in feature_cols:
        valid = features_df[col].notna().sum()
        logger.info(f"  {col}: {valid} valid values")

    # Save features to duckdb
    conn = duckdb.connect(str(DB_PATH))
    conn.execute("DROP TABLE IF EXISTS features_price")
    conn.execute("CREATE TABLE features_price AS SELECT * FROM features_df")
    count = conn.execute("SELECT COUNT(*) FROM features_price").fetchone()[0]
    conn.close()
    logger.success(f"Saved {count} rows to features_price table")


if __name__ == "__main__":
    main()
