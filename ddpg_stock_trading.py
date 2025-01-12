from stock_trading_env import StockTradingEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import deque
import random
import numpy as np

# Think of the actor as a student who is learning to play chess and the target_actor as the student's coach who provides guidance. The student (actor) tries different strategies in real games, improving over time. Meanwhile, the coach (target_actor) remains calm and consistent, offering guidance based on what they have already mastered.

# Now, imagine if the coach (target_actor) changed their advice as frequently as the student (actor) experimented with new strategies—this would confuse the student and lead to chaotic learning. Instead, the coach updates their guidance slowly and smoothly, so the student can benefit from stable feedback.
      

# PolicyNetwork
# returns single action, not after probability
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
    
# ValueFunctionNetwork
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# gamma 0.99?
class DeepDeterministicPolicyGradient:
    def __init__(self, env, gamma=0.95, actor_lr=1e-4, critic_lr=1e-3, tau=0.005, buffer_size=100000):
        self.env = env
        self.gamma = gamma
        self.tau = tau  # For soft updates

        self.input_size = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        self.actor = Actor(self.input_size, self.num_actions)
        self.critic = Critic(self.input_size + self.num_actions) # dc aici +

        # Copy weights to target networks
        self.target_actor = Actor(self.input_size, self.num_actions)
        self.target_critic = Critic(self.input_size + self.num_actions)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def soft_update(self, target, source, tau): # update for target_actor
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


    def train(self, episodes, max_steps, batch_size=64):
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward, steps, done = 0, 0, False

            while (not done) and steps < max_steps:
                state_tensor = torch.FloatTensor(state)

                # 1. actor(st) => at
                # 2. at => s(t+1)
                # 3. target_actor(s(t+1)) => a'(t+1)
                # 4. rt + gamma * Q(s(t+1), a'(t+1)) => Qtarget(st, at)
                # 5. Compare Qtarget(st, at) with Q(st, at)

                # 1. actor(st) => at
                action = self.actor(state_tensor).detach().numpy()
                action += np.random.normal(0, 0.1, size=self.num_actions)  # add Gaussian noise for exploration
                
                # 2. at => s(t+1)
                next_state, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.add((state, action, reward, next_state, done))

                if self.replay_buffer.size() > batch_size: 
                    # pick samples for using
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

                    # 3. target_actor(s(t+1)) => a'(t+1) (but for all next states picked)
                    next_actions = self.target_actor(next_states)

                    # 4. rt + gamma * Q(s(t+1), a'(t+1)) => Qtarget(st, at)
                    next_states_and_actions = torch.cat([next_states, next_actions], dim=1)
                    target_q_values = self.target_critic(next_states_and_actions)
                    td_targets = rewards + self.gamma * target_q_values * (1 - dones)
                    
                    # get Q(st, at)
                    states_and_actions = torch.cat([states, actions], dim=1)
                    q_values = self.critic(states_and_actions)

                    # 5. Compare Qtarget(st, at) with Q(st, at)
                    critic_loss = F.mse_loss(q_values, td_targets.detach())

                    # Update critic
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    # Update actor 
                    states_and_actions = torch.cat([states, self.actor(states)], dim=1)
                    actor_loss = -self.critic(states_and_actions).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Soft updates for target networks
                    self.soft_update(self.target_actor, self.actor, self.tau)
                    self.soft_update(self.target_critic, self.critic, self.tau)
                
                state = next_state
                total_reward += reward

                if done:
                    break

            print(f"Episode {episode}: Total Reward = {total_reward}")
    
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
    
class DeepDeterministicPolicyGradientStableBaseline:
    def __init__(self, env, total_timesteps):
        self.model = DDPG("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, obs):
       action, _ = self.model.predict(obs)
       return action

