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
    "model_name": "stable3",
    "save_name": "stable4",
    "validation": True,
    #
    "runs": 555,
    "epochs": 15,
    "epoch_size": 500,
    #
    "window_size": 20,
    "positions_min": -1,
    "positions_max": 1,
    "positions_count": 19,
    #
    "direction_reward": 1.5,
    "profit_multiplier": 4.2,
    "hold_penalty": -0.2,
    "min_profit": -0.5,
    "position_change_reward": 0.15,
    "hold_threshold": 7,
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


def percentage_increase(a, b):
    if a == 0:
        raise ValueError("The value of 'a' must not be zero.")
    increase = b - a
    percentage = (increase / a) * 100
    return round(percentage, 2)


def get_env(df: pd.DataFrame, positions: list, portfolio_initial_value) -> gym.Env:

    def reward_function(history):
        if len(history["position_index"]) < max(
            hyperparams["hold_threshold"], hyperparams["profit_lookback"]
        ):
            return hyperparams["no_history_penalty"]

        current_valuation = history["portfolio_valuation"][-1]
        previous_valuation = history["portfolio_valuation"][
            -hyperparams["profit_lookback"]
        ]
        current_position = history["position_index"][-1]
        previous_position = (
            history["position_index"][-2] if len(history["position_index"]) > 1 else 0
        )

        profit_reward = (current_valuation / previous_valuation) - 1

        position_change_reward = 0
        if current_position != previous_position:
            position_change_reward = hyperparams["position_change_reward"]

        hold_duration = 0
        for i in range(2, len(history["position_index"])):
            if history["position_index"][-i] == current_position:
                hold_duration += 1
                if hold_duration > hyperparams["hold_threshold"]:
                    break
            else:
                break

        hold_penalty = (
            hyperparams["hold_penalty"] * hold_duration
            if hold_duration > hyperparams["hold_threshold"]
            else 0
        )

        total_reward = profit_reward + position_change_reward + hold_penalty

        if profit_reward > 0:
            total_reward *= hyperparams["profit_multiplier"]
        else:
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
        verbose=0,
        initial_position=0,
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
            learning_rate=0.001,
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
        # print("Validation")
        env2 = get_env(df, positions, hyperparams["portfolio_initial_value"])
        env2.reset()
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            a, b, c, d, info = env2.step(action[0])
            if done:
                print("done")
                env2.unwrapped.save_for_render(dir="render_stable")
                return percentage_increase(
                    hyperparams["portfolio_initial_value"],
                    info["portfolio_valuation"],
                )


arr = []
for i in range(hyperparams["runs"]):
    # print("run ", i)
    arr.append(run())


average_profit = np.mean(arr)
median_profit = np.median(arr)

print(f"Average Profit: {average_profit}%")
print(f"Median Profit: {median_profit}%")
print(arr)
import matplotlib.pyplot as plt

plt.hist(arr, bins=50, edgecolor="black")
plt.title("Distribution of Profits as Percentages")
plt.xlabel("Percentage Increase")
plt.ylabel("Frequency")
plt.show()
