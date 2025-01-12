from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stock_trading_visualizer import StockTradingVisualizer
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_data_pipeline import StockDataPipeline
from single_stock_env.single_stock_trading_env import SingleStockTradingEnv
from single_stock_env.dqn import DQNStableBaseline
from datetime import datetime, timedelta
from ddpg_stock_trading import DeepDeterministicPolicyGradient

import json

EPOCHS = 50
MAX_STEPS = 2000

def load_config(config_path="config.json"):
    with open(config_path, "r") as file:
        return json.load(file)

def prepare_stock_data(tickers, start_date, end_date):
    pipeline = StockDataPipeline(tickers=tickers, start_date=start_date, end_date=end_date, save_processed=True)
    return pipeline.run()

def create_env(stock_data, target_ticker, initial_balance):
    return DummyVecEnv([lambda: SingleStockTradingEnv(stock_data=stock_data, target_ticker=target_ticker, initial_balance=initial_balance)])

def test_agent(env, agent, stock_data, n_tests=1000, visualize=False):
    metrics = {
        'steps': [],
        'balances': [],
        'net_worths': [],
        'shares_held': {config['target_ticker']: []}
    }

    obs = env.reset()
    
    for step in range(n_tests):
        metrics['steps'].append(step)
        action = agent.predict(obs)
        obs, rewards, done, info = env.step(action)

        if visualize:
            env.render()

        balances = env.get_attr('balance')[0]
        net_worths = env.get_attr('net_worth')[0]
        shares_held = env.get_attr('shares_held')[0]

        metrics['balances'].append(balances)
        metrics['net_worths'].append(net_worths)
        metrics['shares_held'][config['target_ticker']].append(shares_held)

        if done:
            obs = env.reset()
            
    return metrics

def test_and_visualize_agents(agents, training_data, n_tests):
    metrics = {}

    for agent_name, agent in agents.items():
        print(f"Testing {agent_name}...")
        env = create_env(training_data, config["target_ticker"], config["initial_balance"])
        metrics[agent_name] = test_agent(env, agent, training_data, n_tests=n_tests, visualize=True)
        print(f"Done testing {agent_name}!")

    visualize_results(metrics)

def visualize_results(metrics):
    steps = next(iter(metrics.values()))['steps']
    net_worths = [metrics[agent_name]['net_worths'] for agent_name in metrics.keys()]

    StockTradingVisualizer.visualize_multiple_portfolio_net_worth(steps, net_worths, list(metrics.keys()))

def get_agents(stock_data, config, save=False): 
    dqn_env = SingleStockTradingEnv(stock_data=stock_data, target_ticker=config['target_ticker'], initial_balance=config["initial_balance"])
    total_timesteps = stock_data[config['target_ticker']].shape[0] * EPOCHS
    stable_baselines_dqn = DQNStableBaseline(dqn_env, total_timesteps)
    
    if save:
        stable_baselines_dqn.save(f"saved/{config['target_ticker']}_dqn_agent_model")
    
    return {
        "DQN Agent (stable_baselines3)": stable_baselines_dqn
    }

if __name__ == "__main__":
    config = load_config()

    data = prepare_stock_data([config["target_ticker"]], config["start_date"], config["end_date"])

    agents = get_agents(data, config, save=True)
    test_and_visualize_agents(agents, data, config["n_tests"])