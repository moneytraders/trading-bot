import matplotlib.pyplot as plt
import numpy as np

class StockTradingVisualizer:
    @staticmethod
    def visualize_portfolio(steps, balances, net_worths, shares_held, tickers, show_balance=True, show_net_worth=True, show_shares_held=True):
        fig, axs = plt.subplots(3, figsize=(12, 18))

        if show_balance:
            axs[0].plot(steps, balances, label='Balance')
            axs[0].set_title('Balance Over Time')
            axs[0].set_xlabel('Steps')
            axs[0].set_ylabel('Balance')
            axs[0].legend()

        if show_net_worth:
            axs[1].plot(steps, net_worths, label='Net Worth', color='orange')
            axs[1].set_title('Net Worth Over Time')
            axs[1].set_xlabel('Steps')
            axs[1].set_ylabel('Net Worth')
            axs[1].legend()

        if show_shares_held:
            for ticker in tickers:
                axs[2].plot(steps, shares_held[ticker], label=f'Shares Held: {ticker}')
            axs[2].set_title('Shares Held Over Time')
            axs[2].set_xlabel('Steps')
            axs[2].set_ylabel('Shares Held')
            axs[2].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_portfolio_net_worth(steps, net_worths):
        plt.figure(figsize=(12, 6))
        plt.plot(steps, net_worths, label='Net Worth', color='orange')
        plt.title('Net Worth Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Net Worth')
        plt.legend()
        plt.show()

    @staticmethod
    def visualize_multiple_portfolio_net_worth(steps, net_worths_list, labels):
        plt.figure(figsize=(12, 6))
        for i, net_worths in enumerate(net_worths_list):
            plt.plot(steps, net_worths, label=labels[i])
        plt.title('Net Worth Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Net Worth')
        plt.legend()
        plt.show()