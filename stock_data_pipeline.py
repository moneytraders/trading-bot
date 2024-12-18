from datetime import datetime, timedelta
from typing import List

import yfinance as yf
import pandas as pd
import numpy as np
import os

class StockDataPipeline:
    def __init__(self, tickers: List[str], start_date: str, end_date: str, save_processed=True):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.save_processed = save_processed
        
        self.train = {}
        self.validation = {}
        self.test = {}
        
        os.makedirs("data", exist_ok=True)
        os.makedirs("processed-data", exist_ok=True)

    def __fetch_data(self):
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            df.reset_index(inplace=True)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.to_csv(f"data/{ticker}.csv", index=False)

    def __split_data(self, train_ratio=0.7, val_ratio=0.2):
        for ticker in self.tickers:
            df = pd.read_csv(f"data/{ticker}.csv", index_col="Date", parse_dates=True)
            df = df.sort_index()
            
            train_split = int(len(df) * train_ratio)
            val_split = int(len(df) * (train_ratio + val_ratio))
            
            self.train[ticker] = df[:train_split]
            self.validation[ticker] = df[train_split:val_split]
            self.test[ticker] = df[val_split:]

    def __calculate_rsi(self, df, window=14):
        """Calculates the Relative Strength Index (RSI)"""
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    def __calculate_macd(self, df):
        """Calculates the Moving Average Convergence Divergence (MACD)"""
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    def __calculate_cci(self, df, window=20, constant=0.015):
        """Calculates the Commodity Channel Index (CCI)"""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=window).mean()
        mean_dev = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (tp - sma_tp) / (constant * mean_dev)

    def __calculate_adx(self, df, window=14):
        """Calculates the Average Directional Index (ADX)"""
        high_diff = df['High'].diff()
        low_diff = df['Low'].diff()
        df['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        df['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        tr = pd.concat([
            df['High'] - df['Low'], 
            np.abs(df['High'] - df['Close'].shift(1)), 
            np.abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=window, adjust=False).mean()
        df['+DI'] = 100 * (df['+DM'].ewm(span=window, adjust=False).mean() / atr)
        df['-DI'] = 100 * (df['-DM'].ewm(span=window, adjust=False).mean() / atr)
        dx = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df['ADX'] = dx.ewm(span=window, adjust=False).mean()

    def __add_technical_indicators(self, df):
        for add_indicator_method in [
            self.__calculate_rsi,
            self.__calculate_macd,
            self.__calculate_cci,
            self.__calculate_adx
        ]:
            add_indicator_method(df)
        
        df.dropna(inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'Signal', 'RSI', 'CCI', 'ADX']]

    def __process_data(self):
        for ticker in self.tickers:
            self.train[ticker] = self.__add_technical_indicators(self.train[ticker])
            self.validation[ticker] = self.__add_technical_indicators(self.validation[ticker])
            self.test[ticker] = self.__add_technical_indicators(self.test[ticker])

        if self.save_processed:
            self.__save_data()

    def __save_data(self):
        for ticker, df in self.train.items():
            df.to_csv(f"processed-data/{ticker}_train.csv", index=False)
        for ticker, df in self.validation.items():
            df.to_csv(f"processed-data/{ticker}_validation.csv", index=False)
        for ticker, df in self.test.items():
            df.to_csv(f"processed-data/{ticker}_test.csv", index=False)

    def run(self):
        self.__fetch_data()
        self.__split_data()
        self.__process_data()
        
        return self.train, self.validation, self.test