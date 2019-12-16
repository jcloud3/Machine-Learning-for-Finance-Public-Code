import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import MarketSimulator as ms
from util import get_data
import TechnicalIndicators as id


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, bb_threshold=1, sma_threshold=0.06, momentum_threshold = 0.4):
    '''
    Generates a dataframe of trades based on the manual strategy policy.
	The policy is described in the attached report.
	
	Parameters:
	symbol (string) - The symbol with which to trade.
	sd (datetime) - Trading period start date.
	ed (datetime) - Trading period end date.
	sv (int) - Starting cash value.
	bb_threshold (float) - Threshold at which to trigger indicator.
	sma_threshold (float) - Threshold at which to trigger indicator.
	momentum_threshold(float) - Threshold at which to trigger indicator.
    '''
    window = 10;
    prices = id.get_prices([symbol], sd, ed)
    indicators = id.get_indicators(prices, [symbol], window = window)

    position = 0;
    holdings = 0;

    df_trades = pd.DataFrame(index = indicators.index);
    df_trades["Symbol"] = symbol
    df_trades["Order"] = "NONE"
    df_trades["Shares"] = 0

    for day in indicators.iloc[window-1:].index:
        bb = indicators["Normal BB"][day]
        sma = indicators["Price/SMA"][day]
        momentum = indicators["Momentum"][day]

        # Short Positions:
        # Stock is above upper Bollinger Band 
        # Price/SMA ratio is above 1.
        # Momentum is smaller than lower threshold
        if (bb > bb_threshold or sma > 1+sma_threshold) or momentum < -momentum_threshold:
            position = -1
        # Long Positions:
        # Price is below lower Bollinger Band
        # Price/SMA ratio is below 1.
        # Momentum is larger than upper threshold
        elif (bb < -bb_threshold or sma < 1-sma_threshold) or momentum > momentum_threshold:
            position = 1
        # Otherwise, maintain position.
        else:
            position = 0

        # Calculate needed shares to trade based on position.
        if position == -1:
            shares_to_trade = -1000-holdings
            holdings = -1000
        elif position == 1:
            shares_to_trade = 1000-holdings
            holdings = 1000
        else:
            shares_to_trade = 0

        if shares_to_trade<0:
            df_trades.loc[day, "Order"] = "SELL"
            df_trades.loc[day, "Shares"]= -shares_to_trade
        elif shares_to_trade>0:
            df_trades.loc[day, "Order"] = "BUY"
            df_trades.loc[day, "Shares"] = shares_to_trade

    df_trades = df_trades[df_trades.Order != "NONE"]
    return df_trades


def main_manual(sd= dt.datetime(2008, 1, 2), ed= dt.datetime(2009, 12, 31), out_file = "default.png", out_subtitle = "Default"):
	'''
	Main script to run an experiment with the manual strategy in comparison to buy and hold benchmark.
	'''
    # Set-up initial parameters for simulation
    symbol = "JPM"

    sv = 100000
    commission = 9.95;
    impact = 0.005;

    # Calculate optimal portfolio values
    manual_trades = testPolicy(symbol = symbol, sd = sd, ed = ed, sv = sv)
    manual_port_vals = ms.compute_portvals(manual_trades, start_val = sv, commission = commission, impact = impact)
    manual_port_vals = manual_port_vals / manual_port_vals.iloc[0,:]
    manual_port_vals.rename(columns = {"Portfolio Value":"Manual"}, inplace = True)

    # Calculate benchmark portfolio values
    benchmark_trades = pd.DataFrame(data = [["JPM","BUY",1000]], index = [sd, ed], columns = ["Symbol", "Order", "Shares"])
    benchmark_port_vals = ms.compute_portvals(benchmark_trades, start_val = sv, commission = commission, impact = impact)
    benchmark_port_vals = benchmark_port_vals / benchmark_port_vals.iloc[0,:]
    benchmark_port_vals.rename(columns = {"Portfolio Value":"Benchmark"}, inplace = True)

    port_vals = pd.DataFrame(benchmark_port_vals["Benchmark"], index = benchmark_port_vals.index)
    port_vals["Manual"] = manual_port_vals["Manual"]
    port_vals.fillna(method = 'ffill', inplace=True)

    # Plot optimal strategy
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.plot(port_vals["Benchmark"], color = 'g', linewidth = 0.7)
    plt.plot(port_vals["Manual"], color = 'r', linewidth = 0.7)
    if out_file == "Manual_In.png":
        print("In-sample")
        for day in manual_trades.index:
            if manual_trades.loc[day, "Order"] == "BUY":
                plt.axvline(day, color = 'b')
            else:
                plt.axvline(day, color = 'k')
    plt.legend(["Benchmark", "Manual Strategy"])
    plt.title("Portfolio Values: Benchmark vs. Manual Strategy\n"+out_subtitle, fontsize = 10)
    plt.ylabel("Normalized Values", fontsize = 8)
    plt.xlabel("Date Range", fontsize = 8)
    plt.xticks(rotation = 45)
    plt.savefig(out_file)
    #plt.show()

    mcr, madr, msddr, msr = id.calculate_portfolio_metrics(manual_port_vals)
    bcr, badr, bsddr, bsr = id.calculate_portfolio_metrics(benchmark_port_vals)
    print("Manual Strategy: ",  mcr, madr, msddr, msr)
    print("Benchmark:", bcr, badr, bsddr, bsr)

if __name__ == "__main__":
    main_manual(out_file = "Manual_In.png", out_subtitle = "In Sample")
    plt.figure()
    main_manual(sd = dt.datetime(2010, 1, 4), ed = dt.datetime(2011, 12, 30), out_file = "Manual_Out.png", out_subtitle = "Out of Sample")
