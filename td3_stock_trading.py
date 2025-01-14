
from stock_trading_env import StockTradingEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import TD3

import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import deque
import random
import numpy as np


class Actor(nn.Module):
    def __init__(self, input_size, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Actions in range [-1, 1]
        return action

class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TwinDelayedDDPG:
    def __init__(self, env, gamma=0.95, tau=0.005, actor_lr=1e-4, critic_lr=1e-3, noise_std=0.2, noise_clip=0.5, policy_delay=2, buffer_size=100000):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay

        self.input_size = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        # Initialize networks
        self.actor = Actor(self.input_size, self.num_actions)
        self.actor_target = Actor(self.input_size, self.num_actions)
        self.critic1 = Critic(self.input_size + self.num_actions)
        self.critic2 = Critic(self.input_size + self.num_actions)
        self.critic1_target = Critic(self.input_size + self.num_actions)
        self.critic2_target = Critic(self.input_size + self.num_actions)

        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Action noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip

    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

    def train(self, episodes, max_steps, batch_size=64):
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward, steps = 0, 0

            while steps < max_steps:
                # Generate action with exploration noise
                state_tensor = torch.FloatTensor(state)
                action = self.actor(state_tensor).detach().numpy()
                action += np.random.normal(0, self.noise_std, size=self.num_actions)
                action = np.clip(action, -1, 1)  # Clip to valid action range

                # Interact with the environment
                next_state, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.add((state, action, reward, next_state, done))

                # Update networks if enough samples are in the replay buffer
                if self.replay_buffer.size() > batch_size:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

                    # Add noise to target actions and clip
                    noise = torch.clamp(
                        torch.normal(0, self.noise_std, size=actions.shape), -self.noise_clip, self.noise_clip
                    )
                    next_actions = self.actor_target(next_states) + noise
                    next_actions = torch.clamp(next_actions, -1, 1)

                    # Calculate target Q-values
                    next_q1 = self.critic1_target(torch.cat([next_states, next_actions], dim=1))
                    next_q2 = self.critic2_target(torch.cat([next_states, next_actions], dim=1))
                    target_q = rewards + self.gamma * (1 - dones) * torch.min(next_q1, next_q2)

                    # Update Critic networks
                    current_q1 = self.critic1(torch.cat([states, actions], dim=1))
                    current_q2 = self.critic2(torch.cat([states, actions], dim=1))
                    critic1_loss = F.mse_loss(current_q1, target_q.detach())
                    critic2_loss = F.mse_loss(current_q2, target_q.detach())

                    self.critic1_optimizer.zero_grad()
                    critic1_loss.backward()
                    self.critic1_optimizer.step()

                    self.critic2_optimizer.zero_grad()
                    critic2_loss.backward()
                    self.critic2_optimizer.step()

                    # Delayed Actor update
                    if steps % self.policy_delay == 0:
                        actor_loss = -self.critic1(torch.cat([states, self.actor(states)], dim=1)).mean()
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        # Soft update target networks
                        self.soft_update(self.actor_target, self.actor, self.tau)
                        self.soft_update(self.critic1_target, self.critic1, self.tau)
                        self.soft_update(self.critic2_target, self.critic2, self.tau)

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

            print(f"Episode {episode}, Total Reward: {total_reward}")

    def predict(self, observation):
        state_tensor = torch.FloatTensor(observation)
        action = self.actor(state_tensor)
        return action.detach().numpy()

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )
    
    def size(self):
        return len(self.buffer)
    

class TwinDelayedDDPGStableBaseline:
    def __init__(self, env, total_timesteps, device="cuda", learn=True):
        self.model = TD3("MlpPolicy", env, verbose=1, device = device)
        if learn:
            self.model.learn(total_timesteps=total_timesteps)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action