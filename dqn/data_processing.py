import pandas as pd
import numpy as np
import talib


def load_data(path: str) -> pd.DataFrame:
    """
    Load and preprocess trading data.
    """
    df = pd.read_csv(path, header=0)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)

    df = df.sort_index().dropna().drop_duplicates()

    # Standardize volume and number of trades
    df["feature_volume"] = (df['volume'] -
                            df['volume'].mean()) / df['volume'].std()
    df["feature_trades"] = (df['nrOfTrades'] -
                            df['nrOfTrades'].mean()) / df['nrOfTrades'].std()

    # Log returns for prices
    df["feature_close_log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["feature_high_log_return"] = np.log(df["high"] / df["high"].shift(1))
    df["feature_low_log_return"] = np.log(df["low"] / df["low"].shift(1))
    df["feature_open_log_return"] = np.log(df["open"] / df["open"].shift(1))

    # Rolling mean and standard deviation for volatility and trend analysis
    df["feature_rolling_mean_30"] = df["close"].rolling(window=30).mean()
    df["feature_rolling_std_30"] = df["close"].rolling(window=30).std()
    df["feature_rolling_mean_60"] = df["close"].rolling(window=60).mean()
    df["feature_rolling_std_60"] = df["close"].rolling(window=60).std()

    # Lagged features for trend prediction
    df["feature_lag_close"] = df["close"].shift(1)
    df["feature_lag_high"] = df["high"].shift(1)
    df["feature_lag_low"] = df["low"].shift(1)
    df["feature_lag_open"] = df["open"].shift(1)

    # TA-Lib indicators
    df['feature_macd'], df['feature_macd_signal'], _ = talib.MACD(df['close'])
    df['feature_rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['feature_upperband'], df['feature_middleband'], df[
        'feature_lowerband'] = talib.BBANDS(df['close'], timeperiod=20)
    df['feature_atr'] = talib.ATR(df['high'],
                                  df['low'],
                                  df['close'],
                                  timeperiod=14)

    df = df.dropna().drop_duplicates()
    return df
