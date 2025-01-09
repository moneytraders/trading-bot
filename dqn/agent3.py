import gymnasium as gym
import gym_trading_env
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from exp_replay import ReplayMemory
from dqn import DQN
import itertools
import yaml
from torchsummary import summary
import talib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


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


def get_env(df: pd.DataFrame, positions: list) -> gym.Env:
    """
    Get the trading environment.
    """

    def reward_function(history):
        return np.log(history["portfolio_valuation", -1] /
                      history["portfolio_valuation", -2])

    env = gym.make(
        "TradingEnv",
        name="SOLUSDT",
        df=df,
        positions=positions,
        trading_fees=0.01 / 100,  # 0.01% per stock buy/sell (Binance fees)
        borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep
        windows=yaml.safe_load(open("hyperparameters.yaml",
                                    'r'))["trading"]["window_size"],
        reward_function=reward_function,
    )
    return env


class DQNAgent:
    """
    A class for the DQN agent.
    """

    def __init__(self):
        hyperparameters = yaml.safe_load(open("hyperparameters.yaml",
                                              'r'))["trading"]
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.minibatch_size = hyperparameters["minibatch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.learning_rate_optimizer = hyperparameters[
            "learning_rate_optimizer"]

        self.positions_min = hyperparameters["positions_min"]
        self.positions_max = hyperparameters["positions_max"]
        self.positions_count = hyperparameters["positions_count"]
        self.enable_double_dqn = hyperparameters["enable_double_dqn"]
        self.window_size = hyperparameters["window_size"]
        self.optimize_every_steps = hyperparameters["optimize_every_steps"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def setup(self, is_training=True):
        """
        Setup the environment and model components.
        """
        df = load_data("SOLUSDT/buckets/SOLUSDT-1m-0.csv")
        positions = np.linspace(self.positions_min, self.positions_max,
                                self.positions_count)
        positions = np.sort(np.insert(positions, 0, 0)).tolist()

        env = get_env(df, positions)
        num_actions = env.action_space.n
        num_observations = env.observation_space.shape[1]

        # Define DQN models
        policy_dqn = DQN(self.window_size, num_observations,
                         num_actions).to(device)
        target_dqn = DQN(self.window_size, num_observations,
                         num_actions).to(device)

        summary(policy_dqn, (self.window_size, num_observations))

        target_dqn.load_state_dict(policy_dqn.state_dict())

        if is_training:
            self.optimizer = optim.Adam(policy_dqn.parameters(),
                                        lr=self.learning_rate_optimizer)
            return env, policy_dqn, target_dqn
        else:
            return env, policy_dqn, None

    def run(self, is_training: bool = True):
        """
        Run the training or evaluation process.
        """
        env, policy_dqn, target_dqn = self.setup(is_training)

        epsilon = self.epsilon_init
        memory = ReplayMemory(self.replay_memory_size) if is_training else None
        epsilon_history, rewards_per_episode = [], []

        for episode in itertools.count():
            print('Episode:', episode)
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            terminated = False
            truncated = False
            episode_reward = 0
            step = 0
            while not terminated and not truncated:
                step += 1
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(
                            state.unsqueeze(0).to(device)).squeeze().argmax()

                new_state, reward, terminated, truncated, new_info = env.step(
                    action.item())
                new_state = torch.tensor(new_state,
                                         dtype=torch.float32).to(device)

                episode_reward += reward

                if is_training:
                    memory.append(
                        (state, action, reward, new_state, terminated))

                    if len(
                            memory
                    ) >= self.minibatch_size and step % self.optimize_every_steps == 0:
                        minibatch = memory.sample(self.minibatch_size)
                        self.optimize(minibatch, policy_dqn, target_dqn)

                state = new_state

                # Sync target DQN
                if is_training and step % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            # Epsilon decay
            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

            epsilon_history.append(epsilon)
            rewards_per_episode.append(episode_reward)
            print(
                f"Episode {episode} | Reward: {episode_reward} | Epsilon: {epsilon}"
            )

    def optimize(self, minibatch, policy_dqn, target_dqn):
        """
        Optimize the policy network using a minibatch from replay memory.
        """
        states, actions, rewards, new_states, terminated = zip(*minibatch)
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions).to(device)
        new_states = torch.stack(new_states).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        terminated = torch.tensor(terminated, dtype=torch.float32).to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (
                    1 - terminated
                ) * self.discount_factor * target_dqn(new_states).gather(
                    dim=1, index=best_actions.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (
                    1 - terminated
                ) * self.discount_factor * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(
            dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    agent = DQNAgent()
    agent.run(True)
