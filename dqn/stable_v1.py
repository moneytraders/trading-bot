from stable_baselines3 import DQN
import torch.nn as nn
import gymnasium as gym
import pandas as pd
import numpy as np
from data_processing import load_data
import gym_trading_env
from stable_baselines3.common.vec_env import DummyVecEnv

import warnings

warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

hyperparams = {
    "model_name": "stable5",
    "save_name": "stable5",
    "validation": True,
    #
    "runs": 5,
    "epochs": 20,
    "epoch_size": 400,
    #
    "window_size": 20,
    "positions_min": -1,
    "positions_max": 1,
    "positions_count": 19,
    #
    "direction_reward": 2.52,
    "profit_multiplier": 450,
    "hold_penalty": -0.05,
    "min_profit": -1,
    "position_change_reward": 0.75,
    "hold_threshold_min": 4,
    "hold_threshold_max": 8,
    "profit_lookback": 5,
    "no_history_penalty": -0.05,
    #
    "borrow_interest_rate": 0.00003,
    "trading_fees": 0.0001,
    "portfolio_initial_value": 10000,
    #
    "batch_size": 128,
    "buffer_size": 3500,
    "learn_log_interval": 1000,
    "epsilon_init": 0.9,
    "epsilon_final": 0.02,
    "learning_rate": 0.001,
}


policy_kwargs = dict(
    activation_fn=nn.ReLU,
    net_arch=[
        256,
        512,
        1024,
        512,
        1024,
        512,
        1024,
        512,
        256,
        512,
        1024,
        512,
        64,
    ],
)

# policy_kwargs = dict(
#     activation_fn=nn.ReLU,
#     net_arch=[256, 512, 1024, 512, 1024, 256, 512, 1024, 512, 64])

positions = np.sort(
    np.unique(  # maybe we have 0 two times
        np.insert(  # we want to insert 0 in the positions list
            np.linspace(
                hyperparams["positions_min"],
                hyperparams["positions_max"],
                hyperparams["positions_count"],
            ),
            0,
            0,
        )
    )
).tolist()


def get_env(df: pd.DataFrame, positions: list, portfolio_initial_value) -> gym.Env:

    def reward_function(history):
        if len(history["position_index"]) < hyperparams["profit_lookback"]:
            return hyperparams["no_history_penalty"]

        current_valuation = history["portfolio_valuation"][-1]
        current_position = history["position_index"][-1]

        previous_position_index = 0
        for i in range(2, len(history["position_index"])):
            if history["position_index"][-i] != current_position:
                previous_position_index = -i
                break

        previous_valuation = history["portfolio_valuation"][previous_position_index]
        previous_position = history["position_index"][previous_position_index]
        previous_price = history["data_close"][previous_position_index]
        last_price = history["data_close"][-1]

        if previous_position_index == 0:
            return -3

        profit_reward = (current_valuation / previous_valuation) - 1
        profit_reward *= hyperparams["profit_multiplier"]

        position_change_reward = 0
        if current_position != previous_position:
            position_change_reward = hyperparams["position_change_reward"]

        hold_duration = (
            len(history["position_index"]) + previous_position_index
        ) % len(history["position_index"])

        hold_penalty = (
            hyperparams["hold_penalty"] * hold_duration
            if hold_duration > hyperparams["hold_threshold_max"]
            or hold_duration < hyperparams["hold_threshold_min"]
            else 0
        )

        price_change = last_price - previous_price

        if previous_position < current_position and price_change > 0:
            direction_reward = hyperparams["direction_reward"]
        elif previous_position > current_position and price_change < 0:
            direction_reward = hyperparams["direction_reward"]
        else:
            direction_reward = hyperparams["min_profit"] / 2

        total_reward = (
            profit_reward + position_change_reward + hold_penalty + direction_reward
        )

        total_reward = max(total_reward, hyperparams["min_profit"])

        return total_reward

    env = gym.make(
        "TradingEnv",
        name="SOLUSDT",
        df=df,
        positions=positions,
        trading_fees=hyperparams["trading_fees"] / 100,
        borrow_interest_rate=hyperparams["borrow_interest_rate"] / 100,
        windows=hyperparams["window_size"],
        reward_function=reward_function,
        portfolio_initial_value=portfolio_initial_value,
        verbose=1,
        initial_position=0,
        # dynamic_feature_functions=[],
    )
    return env


def get_df():
    if hyperparams["validation"]:
        bucket = np.random.randint(0, 30)
        # bucket = np.random.randint(0, 22) * 2
    else:
        # bucket = np.random.randint(0, 22) * 2 + 1
        bucket = np.random.randint(31, 44)

    df = load_data(f"../../../SOLUSDT/buckets5/SOLUSDT-1m-{bucket}.csv")
    try:
        start_idx = np.random.randint(0, len(df) - hyperparams["epoch_size"])
    except:
        start_idx = 0

    df = df.iloc[start_idx : min(start_idx + hyperparams["epoch_size"], len(df))]
    return df


def run():
    df = get_df()

    env = DummyVecEnv(
        [lambda: get_env(df, positions, hyperparams["portfolio_initial_value"])]
    )

    if hyperparams["model_name"]:
        model = DQN.load(hyperparams["model_name"])
        model.set_env(env)
    else:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=hyperparams["learning_rate"],
            buffer_size=hyperparams["buffer_size"],
            learning_starts=hyperparams["window_size"],
            batch_size=hyperparams["batch_size"],
            train_freq=30,
            policy_kwargs=policy_kwargs,
            exploration_initial_eps=hyperparams["epsilon_init"],
            exploration_final_eps=hyperparams["epsilon_final"],
        )

    if not hyperparams["validation"]:
        print("Training")
        model.learn(
            total_timesteps=hyperparams["epoch_size"] * hyperparams["epochs"],
            log_interval=hyperparams["learn_log_interval"],
        )
        print("Training done")
        model.save(hyperparams["save_name"])
    else:
        print("Validation")
        env2 = get_env(df, positions, hyperparams["portfolio_initial_value"])
        env2.reset()
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            env2.step(action[0])
            if done:
                print("done")
                env2.unwrapped.save_for_render(dir="render_stable")
                break


for i in range(hyperparams["runs"]):
    print("run ", i)
    run()


hyperparams["model_name"] = hyperparams["save_name"]
hyperparams["validation"] = True
run()
