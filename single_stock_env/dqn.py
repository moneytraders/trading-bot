import gym

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3 import DQN

class DQNStableBaseline:
    def __init__(self, env, total_timesteps):
        self.model = DQN("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action
    
    def save(self, path):
        self.model.save(path)