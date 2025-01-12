from gymnasium import spaces
import gymnasium as gym
import pandas as pd
import numpy as np

class SingleStockTradingEnv(gym.Env):
    def __init__(self, stock_data, target_ticker, initial_balance=10000, transaction_cost_percent=0.005):
        super(SingleStockTradingEnv, self).__init__()
        
        self.stock_data = {ticker: df for ticker, df in stock_data.items() if not df.empty}
        self.tickers = list(self.stock_data.keys())
        
        if not self.tickers:
            raise ValueError("All provided stock data is empty")
        
        if target_ticker not in self.tickers:
            raise ValueError(f"Target ticker {target_ticker} not found in stock data")
        self.ticker = target_ticker
        sample_df = self.stock_data[self.ticker]
        self.n_features = len(sample_df.columns)
        
        # Discrete action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        """ Observation space:
            - "self.n_features" = Each stock's data 
            - "2" = account balance and net worth
            - "1" = number of shares held for the stock
            - "2" = maximum net worth and current step
        """
        self.obs_shape = self.n_features + 2 + 1 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)
        
        # Initialize account balance
        self.initial_balance = initial_balance
        self.transaction_cost_percent = transaction_cost_percent
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state of the environment to an initial state
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        return self._next_observation(), {}

    def _next_observation(self):
        # Ensure current_step is within bounds
        if self.current_step >= len(self.stock_data[self.ticker]):
            self.current_step = len(self.stock_data[self.ticker]) - 1

        # Get the next observation
        obs = list(self.stock_data[self.ticker].iloc[self.current_step].values)
        
        obs.extend([self.balance, self.net_worth])
        obs.append(self.shares_held)
        obs.extend([self.max_net_worth, self.current_step])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        if self.current_step >= len(self.stock_data[self.ticker]):
            done = True
        else:
            done = False

        obs = self._next_observation()
        if action != 1 and action != 2:
            reward = 0
        else:
            reward = self.net_worth - self.initial_balance
            
        return obs, reward, done, False, {'net_worth': self.net_worth}

    def _take_action(self, action):
        # Take an action in the environment
        if action == 1:  # Buy
            self._buy_stock()
        elif action == 2:  # Sell
            self._sell_stock()

    def _buy_stock(self):
        # Buy stock
        available_balance = self.balance
        self.balance -= available_balance
        self.shares_held += available_balance / self.stock_data[self.ticker].iloc[self.current_step]['Close']
        self.net_worth = self.balance + self.shares_held * self.stock_data[self.ticker].iloc[self.current_step]['Close']

    def _sell_stock(self):
        # Sell stock
        self.balance += self.shares_held * self.stock_data[self.ticker].iloc[self.current_step]['Close']
        self.total_shares_sold += self.shares_held
        self.total_sales_value += self.shares_held * self.stock_data[self.ticker].iloc[self.current_step]['Close']
        self.shares_held = 0
        self.net_worth = self.balance

    def render(self, mode='human'):
        # Render the environment to the screen
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'{self.ticker} Shares held: {self.shares_held}')
        print(f'Net worth: {self.net_worth:.2f}')
        print(f'Profit: {profit:.2f}')
        print(f'Total shares sold: {self.total_shares_sold}')
        print(f'Total sales value: {self.total_sales_value:.2f}')
        print("-----------------------------------------")