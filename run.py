from stock_trading_visualizer import StockTradingVisualizer
from stable_baselines3.common.vec_env import DummyVecEnv
from a2c_stock_trading import AdvantageActorCritic
from stock_data_pipeline import StockDataPipeline
from stock_trading_env import StockTradingEnv
from datetime import datetime, timedelta

import json

def load_config(config_path="config.json"):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

def prepare_stock_data(tickers, lookback_years):
    start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    pipeline = StockDataPipeline(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        save_processed=True,
    )
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

    for i in range(n_tests):
        metrics['steps'].append(i)
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
            if ticker in shares_held:
                metrics['shares_held'][ticker].append(shares_held[ticker])
            else:
                metrics['shares_held'][ticker].append(0)

        if done:
            obs = env.reset()
            
    return metrics

def train_and_evaluate_agents(env, agents, training_data, n_tests):
    metrics = {}

    for agent_name, agent in agents.items():
        print(f"Testing {agent_name}...")
        metrics[agent_name] = test_agent(env, agent, training_data, n_tests=n_tests, visualize=True)
        print(f"Done testing {agent_name}!")

    visualize_results(metrics)

def visualize_results(metrics):
    steps = next(iter(metrics.values()))['steps']
    net_worths = [metrics[agent_name]['net_worths'] for agent_name in metrics.keys()]

    StockTradingVisualizer.visualize_multiple_portfolio_net_worth(steps, net_worths, list(metrics.keys()))

if __name__ == "__main__":
    config = load_config()

    training_data, validation_data, testing_data = prepare_stock_data(config["tickers"], config["lookback_years"])
    env = create_env(training_data, config["initial_balance"])

    agents = {
        "A2C Agent": AdvantageActorCritic(env)
    }

    train_and_evaluate_agents(env, agents, training_data, config["n_tests"])
