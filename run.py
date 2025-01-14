from ppo_stock_trading import ProximalPolicyOptimization, ProximalPolicyOptimizationStableBaseline
from a2c_stock_trading import AdvantageActorCritic, AdvantageActorCriticStableBaseline
from ddpg_stock_trading import DeepDeterministicPolicyGradient, DeepDeterministicPolicyGradientStableBaseline
from td3_stock_trading import TwinDelayedDDPG, TwinDelayedDDPGStableBaseline
from stock_trading_visualizer import StockTradingVisualizer
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_data_pipeline import StockDataPipeline
from stock_trading_env import StockTradingEnv
from datetime import datetime, timedelta

import json


EPOCHS = 50
MAX_STEPS = 2000

def load_config(config_path="config.json"):
    with open(config_path, "r") as file:
        return json.load(file)

def prepare_stock_data(tickers, start_date, end_date):
    pipeline = StockDataPipeline(tickers=tickers, start_date=start_date, end_date=end_date, save_processed=True)
    return pipeline.run()

def create_env(stock_data, initial_balance):
    return DummyVecEnv([lambda: StockTradingEnv(stock_data=stock_data, initial_balance=initial_balance)])

def test_agent(env, agent, stock_data, n_tests=1000, visualize=False):
    metrics = {
        'steps': [],
        'balances': [],
        'net_worths': [],
        'shares_held': {ticker: [] for ticker in stock_data.keys()}
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

        for ticker in stock_data.keys():
            metrics['shares_held'][ticker].append(shares_held.get(ticker, 0))

        if done:
            obs = env.reset()
            
    return metrics

def test_and_visualize_agents(agents, training_data, n_tests):
    metrics = {}

    for agent_name, agent in agents.items():
        print(f"Testing {agent_name}...")
        env = create_env(training_data, config["initial_balance"])
        metrics[agent_name] = test_agent(env, agent, training_data, n_tests=n_tests, visualize=True)
        print(f"Done testing {agent_name}!")

    visualize_results(metrics)

def visualize_results(metrics):
    steps = next(iter(metrics.values()))['steps']
    net_worths = [metrics[agent_name]['net_worths'] for agent_name in metrics.keys()]

    StockTradingVisualizer.visualize_multiple_portfolio_net_worth(steps, net_worths, list(metrics.keys()))

def get_agents(stock_data, config): 
    ppo = ProximalPolicyOptimization(StockTradingEnv(stock_data=stock_data, initial_balance=config["initial_balance"]))
    ppo.train(EPOCHS, MAX_STEPS)
    
    a2c = AdvantageActorCritic(StockTradingEnv(stock_data=stock_data, initial_balance=config["initial_balance"]))
    a2c.train(EPOCHS, MAX_STEPS)

    ddpg = DeepDeterministicPolicyGradient(StockTradingEnv(stock_data=stock_data, initial_balance=config["initial_balance"]))
    ddpg.train(EPOCHS, MAX_STEPS)

    td3 = TwinDelayedDDPG(StockTradingEnv(stock_data=stock_data, initial_balance=config["initial_balance"]))
    td3.train(EPOCHS, MAX_STEPS)

    stable_baselines_a2c = AdvantageActorCriticStableBaseline(create_env(stock_data, config["initial_balance"]), MAX_STEPS * EPOCHS)
    stable_baselines_ppo = ProximalPolicyOptimizationStableBaseline(create_env(stock_data, config["initial_balance"]), MAX_STEPS * EPOCHS)
    stable_baselines_ddpg = DeepDeterministicPolicyGradientStableBaseline(create_env(stock_data, config["initial_balance"]), MAX_STEPS * EPOCHS)
    stable_baselines_td3 = TwinDelayedDDPGStableBaseline(create_env(stock_data, config["initial_balance"]), MAX_STEPS * EPOCHS)

    return {
        "A2C Agent": a2c,
        "PPO Agent": ppo,
        "DDPQ Agent": ddpg,
        "TD3 Agent": td3,
        
        "A2C Agent (stable_baselines3)": stable_baselines_a2c,
        "PPO Agent (stable_baselines3)": stable_baselines_ppo,
        "DDPG Agent (stable_baselines3)": stable_baselines_ddpg,
        "TD3 Agent (stable_baselines3)": stable_baselines_td3
    }

if __name__ == "__main__":
    config = load_config()

    data_train = prepare_stock_data(config["tickers"], config["start_date"], config["end_date"])
    data_test = prepare_stock_data(config["tickers"], config["start_val_date"], config["end_val_date"])
    
    agents = get_agents(data_train, config)
    test_and_visualize_agents(agents, data_test, data_test[config["target_ticker"]].shape[0] - 3)
