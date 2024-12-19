from stock_trading_env import StockTradingEnv
from stable_baselines3 import A2C

import torch.nn.functional as F
import torch.nn as nn
import torch

# PolicyNetwork
class Actor(nn.Module):
    def __init__(self, input_size, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mean = nn.Linear(64, num_actions)
        self.fc3_std = nn.Linear(64, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc3_mean(x))
        std = torch.sigmoid(self.fc3_std(x)) + 1e-6
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

class AdvantageActorCritic():
    def __init__(self, env: StockTradingEnv, gamma=0.99, actor_lr=1e-3, critic_lr=1e-3):
        self.env = env
        self.gamma = gamma
        
        self.input_size = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        self.actor  = Actor(self.input_size, self.num_actions)
        self.critic = Critic(self.input_size)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    def train(self, episodes, max_steps):
        for episode in range(0, episodes):
            total_reward, steps, done = 0, 0, False 
            state, _ = self.env.reset()
            
            while (not done) and steps < max_steps:
                state_tensor = torch.FloatTensor(state)
                
                # pi(s) & take action
                mean, std = self.actor(state_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                action = torch.tanh(action) # action between [-1, 1]
                
                next_state, reward, done, _, _ = self.env.step(action.tolist())
                
                # V(s), V(s')
                value = self.critic(state_tensor)
                next_value = self.critic(torch.FloatTensor(next_state))
                
                # TD target & TD advantage
                td_target = reward + self.gamma * next_value * (1 - done)
                advantage = td_target - value
                
                # Update critic
                critic_loss = F.mse_loss(value, td_target.detach())
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                # Update actor
                log_prob = dist.log_prob(action)
                actor_loss = -(log_prob * advantage.detach()).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                state = next_state
                total_reward += reward
                steps += 1
            
            print(f"Episode {episode}: {reward}")
    
    def predict(self, observation):
        state_tensor = torch.FloatTensor(observation)
        mean, std = self.actor(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.tanh(action)
        return action.detach().numpy()

class AdvantageActorCriticStableBaseline:
   def __init__(self, env, total_timesteps):
       self.model = A2C("MlpPolicy", env, verbose=1)
       self.model.learn(total_timesteps=total_timesteps)
  
   def predict(self, obs):
       action, _ = self.model.predict(obs)
       return action