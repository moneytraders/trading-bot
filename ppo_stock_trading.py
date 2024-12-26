from stock_trading_env import StockTradingEnv
from torch.optim.lr_scheduler import StepLR
from dataclasses import dataclass, field
from torch.distributions import Normal
from stable_baselines3 import PPO
from typing import List

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os

@dataclass
class Experience:
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    gaes: List[float] = field(default_factory=list)
    
    def ensure_numpy(self, array):
        if isinstance(array, np.ndarray):
            return array
        return np.array(array, dtype=np.float32)
    
    def to_numpy(self):
        return {
            "observations": self.ensure_numpy(self.observations),
            "actions": self.ensure_numpy(self.actions),
            "rewards": self.ensure_numpy(self.rewards),
            "values": self.ensure_numpy(self.values),
            "returns": self.ensure_numpy(self.returns),
            "log_probs": self.ensure_numpy(self.log_probs),
            "gaes": self.ensure_numpy(self.gaes)
        }

    def to_tensor(self):
        return {
            "observations": torch.tensor(self.ensure_numpy(self.observations), dtype=torch.float32),
            "actions": torch.tensor(self.ensure_numpy(self.actions), dtype=torch.float32),
            "rewards": torch.tensor(self.ensure_numpy(self.rewards), dtype=torch.float32),
            "values": torch.tensor(self.ensure_numpy(self.values), dtype=torch.float32),
            "returns": torch.tensor(self.ensure_numpy(self.returns), dtype=torch.float32),
            "log_probs": torch.tensor(self.ensure_numpy(self.log_probs), dtype=torch.float32),
            "gaes": torch.tensor(self.ensure_numpy(self.gaes), dtype=torch.float32),
        }

# PolicyNetwork
class Actor(nn.Module):
    def __init__(self, input_size, num_actions, clip_min=-20, clip_max=0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mean = nn.Linear(64, num_actions)
        self.fc3_std = nn.Linear(64, num_actions)
        
        for layer in [self.fc1, self.fc2, self.fc3_mean, self.fc3_std]:
            nn.init.orthogonal_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)

        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = torch.tanh(self.fc3_mean(x))
        
        log_std = torch.clamp(self.fc3_std(x), self.clip_min, self.clip_max)
        std = log_std.exp()
        
        return mean, std
  
# ValueFunctionNetwork
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ProximalPolicyOptimization:
    def __init__(self,
                env,
                gamma=0.99,
                lmbda=0.97,
                actor_lr=3e-5,
                critic_lr=3e-5,
                epsilon=0.2,
                k_epochs=10,
                entropy_cf=0.01,
                max_grad_norm=1):
        self.env = env
        self.lmbda = lmbda
        self.gamma = gamma
        self.epsilon = epsilon
        self.k_epochs = k_epochs
        self.entropy_cf = entropy_cf
        self.max_grad_norm = max_grad_norm

        self.input_size = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        self.actor = Actor(self.input_size, self.num_actions)
        self.critic = Critic(self.input_size)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
                
    def train_critic_network(self, observations, returns):
        for _ in range(self.k_epochs):
            values = self.critic(observations).squeeze(-1)
            critic_loss = F.mse_loss(values, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
    def train_actor_network(self, observations, actions, old_log_probabilities, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.k_epochs):        
            mean, std = self.actor(observations)
            dist = torch.distributions.Normal(mean, std)
            new_log_probabilities = dist.log_prob(actions).sum(dim=-1)
            
            probability_ratio = torch.exp(new_log_probabilities - old_log_probabilities)
            
            clipped_probability_ratio = probability_ratio.clamp(1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -torch.min(probability_ratio * advantages, 
                                   clipped_probability_ratio * advantages)
            
            entropy = dist.entropy().sum(dim=-1)    
            policy_loss = (policy_loss - self.entropy_cf * entropy).mean()
            
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

    def compute_discounted_rewards(self, rewards):
        result = np.zeros_like(rewards, dtype=np.float32)
        cumulative_discount = 0.0
        for t in reversed(range(len(rewards))):
            cumulative_discount = cumulative_discount * self.gamma + rewards[t]
            result[t] = cumulative_discount
        return result
    
    def compute_gaes(self, rewards, values):
        next_values = np.concatenate((values[1:], np.zeros(1)))
        deltas = rewards + self.gamma * next_values - values
        
        gaes = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            gaes[t] = deltas[t] + self.gamma * self.lmbda * (gaes[t + 1] if t + 1 < len(rewards) else 0)
        
        return gaes
    
    def collect_experience(self, max_steps):
        observation, _ = self.env.reset()
        experience = Experience()
        cumulative_reward = 0.0

        for step in range(max_steps):
            obs_tensor = torch.FloatTensor(observation)
           
            mean, std = self.actor(obs_tensor)
            value = self.critic(obs_tensor)
            
            dist = torch.distributions.Normal(mean, std)
            action = torch.tanh(dist.sample())
            action_log_prob = dist.log_prob(action).sum(dim=-1).detach().numpy()

            action_np = action.detach().numpy()
            value_np = value.item()

            next_observation, reward, done, _, _ = self.env.step(action_np)

            experience.observations.append(observation)
            experience.actions.append(action_np)
            experience.rewards.append(reward)
            experience.values.append(value_np)
            experience.log_probs.append(action_log_prob)

            cumulative_reward += reward
            observation = next_observation

            if done:
                break
            
        experience.returns = self.compute_discounted_rewards(np.array(experience.rewards))
        experience.gaes = self.compute_gaes(np.array(experience.rewards), np.array(experience.values))

        return experience, cumulative_reward    
   
    def train(self, episodes, max_steps):
        for episode in range(0, episodes):
            experience, cumulative_reward = self.collect_experience(max_steps)
            exp_tensors = experience.to_tensor()

            self.train_actor_network(exp_tensors["observations"], exp_tensors["actions"], exp_tensors["log_probs"], exp_tensors["gaes"])
            self.train_critic_network(exp_tensors["observations"], exp_tensors["returns"])
            
            print(f"Episode {episode}: {cumulative_reward}")
    
    def predict(self, observation):
        state_tensor = torch.FloatTensor(observation)
        mean, std = self.actor(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.tanh(action)
        return action.detach().numpy()
    
class ProximalPolicyOptimizationStableBaseline:
    def __init__(self, env, total_timesteps):
        self.model = PPO("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action