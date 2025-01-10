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
# from torchsummary import summary
from data_processing import load_data, load_hyperparameters
from tools import percentage_increase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def get_env(df: pd.DataFrame, positions: list,
            portfolio_initial_value) -> gym.Env:
    """
    Get the trading environment.
    """

    # def reward_function(history):
    #     return np.log(history["portfolio_valuation", -1] /
    #                   history["portfolio_valuation", -2])

    def reward_function(history, window_size=20):
        valuation_diff = np.log(history["portfolio_valuation", -1] /
                                history["portfolio_valuation", -2])

        position_reward = np.log(
            1 / abs(history["position", -1])) if history["position",
                                                         -1] != 0 else 0

        return valuation_diff + position_reward

    file = open("hyperparameters.yaml", 'r')
    hyperparams = yaml.safe_load(file)["trading"]
    env = gym.make(
        "TradingEnv",
        name="SOLUSDT",
        df=df,
        positions=positions,
        trading_fees=0.01 / 100,  # 0.01% per stock buy/sell (Binance fees)
        borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep
        windows=hyperparams["window_size"],
        reward_function=reward_function,
        portfolio_initial_value=portfolio_initial_value,
        verbose=1,
    )
    file.close()
    return env


class DQNAgent:
    """
    A class for the DQN agent.
    """

    def __init__(self):
        file = open("hyperparameters.yaml", 'r')
        hyperparameters = yaml.safe_load(file)["trading"]
        file.close()
        load_hyperparameters(self, hyperparameters)

        self.loss_fn = nn.MSELoss()

    def setup(self, is_training, model_path):
        """
        Setup the environment and model components.
        """
        df = load_data(
            f"../../../SOLUSDT/buckets/SOLUSDT-1m-{self.data_bucket}.csv")

        if is_training:
            df = df.iloc[:len(df) // 2]
        else:
            df = df.iloc[len(df) // 2:]

        positions = np.linspace(self.positions_min, self.positions_max,
                                self.positions_count)
        positions = np.sort(np.insert(positions, 0, 0)).tolist()

        env = get_env(df, positions, self.portfolio_initial_value)

        num_actions = env.action_space.n  # len(positions)
        num_observations = env.observation_space.shape[
            1]  # at index 0 is the windows size

        policy_dqn = DQN(self.window_size, num_observations,
                         num_actions).to(device)

        if model_path is not None:
            policy_dqn.load_state_dict(
                torch.load(model_path, weights_only=True))

        # summary(policy_dqn, (self.window_size, num_observations))

        if is_training:
            target_dqn = DQN(self.window_size, num_observations,
                             num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            optimizer = optim.Adam(policy_dqn.parameters(),
                                   lr=self.learning_rate_optimizer)

            return env, policy_dqn, target_dqn, optimizer

        return env, policy_dqn, None, None

    def run(self, is_training, model_path):
        """
        Run the training or evaluation process.
        """

        env, policy_dqn, target_dqn, optimizer = self.setup(
            is_training, model_path)

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(
                self.replay_memory_size) if is_training else None
            epsilon_history, rewards_per_episode = [], []

            best_profit = -np.inf
        else:
            print("validation")

        for episode in itertools.count() if is_training else range(1):
            print()
            print(episode)
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
                        self.optimize(optimizer, minibatch, policy_dqn,
                                      target_dqn)

                state = new_state
                info = new_info

                if is_training and step % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            profit = percentage_increase(self.portfolio_initial_value,
                                         info["portfolio_valuation"])
            if is_training:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if profit > best_profit:
                    best_profit = profit
                    torch.save(policy_dqn.state_dict(),
                               f"models/best_model_{profit}.pth")

                    validation_agent = DQNAgent()
                    validation_profit = validation_agent.run(
                        False, f"models/best_model_{profit}.pth")

                    print(
                        f"Profit: {profit}, Validation profit: {validation_profit}"
                    )

                epsilon_history.append(epsilon)
                rewards_per_episode.append(episode_reward)
            else:
                env.unwrapped.save_for_render(dir="render_logs")
                return profit

            # print(f"Episode {episode} | Profit: {profit} | Epsilon: {epsilon}")

    def optimize(self, optimizer, minibatch, policy_dqn, target_dqn):
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    file = open("hyperparameters.yaml", 'r')
    hyperparameters = yaml.safe_load(file)["trading"]
    file.close()

    agent = DQNAgent()
    c = agent.run(hyperparameters["is_training"],
                  hyperparameters["model_path"])
    print(c)
