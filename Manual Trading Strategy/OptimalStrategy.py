import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import MarketSimulator as ms
from util import get_data
import TechnicalIndicators as id

def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):

    # Retrieve prices on the symbol to be traded
    policy_frame = id.get_prices([symbol], sd, ed)
    
    # Calculate daily returns
    policy_frame['Daily Rets'] = id.get_daily_returns(policy_frame)

    # Calculate positions based on the next day's daily returns
    policy_frame['Positions'] = policy_frame['Daily Rets'].shift(-1)
    policy_frame['Positions'] = np.where(policy_frame['Positions']>0, 1, policy_frame['Positions'])
    policy_frame['Positions'] = np.where(policy_frame['Positions']<0, -1, policy_frame['Positions'])

    # Calculate trades based on positions
    policy_frame['Trades'] = policy_frame['Positions'].shift(1)
    policy_frame['Trades'][0] = 0;
    policy_frame['Trades'] = (policy_frame['Positions']-policy_frame['Trades'])*1000
    policy_frame['Trades'][-1] = 0;
    return policy_frame['Trades']

def process_trades_df(df, symbol):
    df_trades = pd.DataFrame(index = df.index);
    df_trades["Symbol"] = symbol
    df_trades["Order"] = df.values
    df_trades["Shares"] = df.values
    df_trades["Order"] = np.where(df_trades["Shares"]>=0, "BUY", "nan")
    df_trades["Order"] = np.where(df_trades["Shares"]<0, "SELL", df_trades["Order"])
    df_trades["Shares"] = df_trades["Shares"].abs()
    return df_trades

def main_optimal():

    # Set-up initial parameters for simulation
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 2)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 0;
    impact = 0;

    # Calculate optimal portfolio values
    optimal_trades = testPolicy(symbol = symbol, sd = sd, ed = ed, sv = sv)
    optimal_trades = process_trades_df(optimal_trades, "JPM")
    optimal_port_vals = ms.compute_portvals(optimal_trades, start_val = sv, commission = commission, impact = impact)
    optimal_port_vals = optimal_port_vals / optimal_port_vals.iloc[0,:]

    # Calculate benchmark portfolio values
    benchmark_trades = pd.DataFrame(data = [["JPM","BUY",1000]], index = [sd, ed], columns = ["Symbol", "Order", "Shares"])
    benchmark_port_vals = ms.compute_portvals(benchmark_trades, start_val = sv, commission = commission, impact = impact)
    benchmark_port_vals = benchmark_port_vals / benchmark_port_vals.iloc[0,:]

    # Plot optimal strategy
    plt.plot(benchmark_port_vals, color = 'g', linewidth = 0.7)
    plt.plot(optimal_port_vals, color = 'r', linewidth = 0.7)
    plt.legend(["Benchmark", "Optimal"])
    plt.title("Portfolio Values: Benchmark vs. Optimal", fontsize = 10)
    plt.ylabel("Normalized Values", fontsize = 8)
    plt.xlabel("Date Range", fontsize = 8)
    plt.savefig('Optimal.png')

    ocr, oadr, osddr, osr = id.calculate_portfolio_metrics(optimal_port_vals)
    bcr, badr, bsddr, bsr = id.calculate_portfolio_metrics(benchmark_port_vals)
    print("Optimal: ",  ocr, oadr, osddr, osr)
    print("Benchmark:", bcr, badr, bsddr, bsr)




def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:]/df[:-1].values)-1
    daily_returns = daily_returns[1:] #Eliminate 0th element b/c value = 0
    return daily_returns

def author():
    return 'sliang76'

if __name__ == "__main__":
    main_optimal();


