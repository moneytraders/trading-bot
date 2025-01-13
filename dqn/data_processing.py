import pandas as pd
import numpy as np
import talib


def load_data(path: str) -> pd.DataFrame:
    """
    Load and preprocess trading data.
    """
    df = pd.read_csv(path, header=0)
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df.set_index("date", inplace=True)

    df = df.sort_index().dropna().drop_duplicates()

    df["feature_close"] = df["close"].pct_change()
    df["feature_high"] = df["high"].pct_change()
    df["feature_low"] = df["low"].pct_change()
    df["feature_open"] = df["open"].pct_change()

    df["feature_ema15"] = talib.EMA(df["close"], timeperiod=15).pct_change()
    df["feature_ema30"] = talib.EMA(df["close"], timeperiod=30).pct_change()
    df["feature_ema60"] = talib.EMA(df["close"], timeperiod=60).pct_change()
    df["feature_ema90"] = talib.EMA(df["close"], timeperiod=90).pct_change()

    df["feature_rsi"] = talib.RSI(df["close"], timeperiod=14) / 100

    df = df.dropna().drop_duplicates()
    return df


def load_hyperparameters(obj, hyperparameters):
    obj.replay_memory_size = hyperparameters["replay_memory_size"]
    obj.minibatch_size = hyperparameters["minibatch_size"]
    obj.epsilon_init = hyperparameters["epsilon_init"]
    obj.epsilon_decay = hyperparameters["epsilon_decay"]
    obj.epsilon_min = hyperparameters["epsilon_min"]
    obj.network_sync_rate = hyperparameters["network_sync_rate"]
    obj.discount_factor = hyperparameters["discount_factor"]
    obj.learning_rate_optimizer = hyperparameters["learning_rate_optimizer"]
    obj.positions_min = hyperparameters["positions_min"]
    obj.positions_max = hyperparameters["positions_max"]
    obj.positions_count = hyperparameters["positions_count"]
    obj.window_size = hyperparameters["window_size"]
    obj.optimize_every_steps = hyperparameters["optimize_every_steps"]
    obj.portfolio_initial_value = hyperparameters["portfolio_initial_value"]
    obj.data_bucket = hyperparameters["data_bucket"]
    obj.is_training = hyperparameters["is_training"]
    obj.model_path = hyperparameters["model_path"]
    obj.render_dir = hyperparameters["render_dir"]
