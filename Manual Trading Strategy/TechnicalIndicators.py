import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math
from util import get_data

def get_daily_returns(prices):
    '''
    Calculate daily returns for the chosen symbols
    
    Parameters:
    prices: list of adjusted close prices
    Returns: dataframe of daily returns between the input date range
    '''
    return prices / prices.shift(1) - 1

def get_prices(symbols, start_date, end_date):
    '''
    Returns prices for symbols between a date range.

    Parameters:
    symbols: list of symbols
    start_date: datetime object for the start date
    end_date: datetime object for the end date

    Returns: dataframe of prices for the input symbols
    '''
    date_range = pd.date_range(start_date, end_date)
    prices = get_data(symbols, date_range)
    prices = prices[symbols] #remove SPY
    prices = prices/prices.iloc[0] #normalize
    return prices

def get_simple_moving_average(prices, window=10):
    '''
    Calculates simple moving average of prices, given specified window size

    Parameters:
    prices: adjusted closing price for symbols
    window: time window in days, default 10

    Returns: Dataframes of SMA, price to SMA ratio
    '''
    sma = prices.rolling(window=window, center=False).mean()
    price_to_sma = prices.div(sma)
    return sma, price_to_sma

def get_standard_deviation(prices, window=10):
    '''
    Calculates standard deviation of the sma of prices, given specified window size

    Parameters:
    prices: adjusted closing price for symbols
    window: time window in days, default 10
    
    Returns: Dataframe of standard deviation of SMA
    '''
    return prices.rolling(window=window, center=False).std()

def get_bollinger_bands(prices, window = 10, band_size = 2):
    '''
    Calculates upper and lower Bollinger bands and deviation of a price from the mean.

    Parameters:
    prices: adjusted closing price for symbols
    window: time window in days, default 10
    band_size: size of the band (number of standard deviations)

    Returns:
    upper_bb_band: Dataframe of upper band
    lower_bb_band: Datafram of lower band
    bb_value: Deviations from mean.
    '''
    sma, _ = get_simple_moving_average(prices, window = window)
    std = get_standard_deviation(prices, window = window)
    upper_bb_band = sma + std*band_size
    lower_bb_band = sma - std*band_size
    bb_value = (prices-sma)/(band_size*std)
    return upper_bb_band, lower_bb_band, bb_value

def get_momentum(prices, window=10):
    '''
    Calculates momentum.

    Parameters:
    prices: adjusted closing price for symbols
    window: time window in days, default 10

    Returns: momentum dataframe.
    '''
    return prices/prices.shift(window-1)-1

def get_indicators(prices, symbols, window=10):
    '''
    Compiles all indicators into one dataframe for easy access.

    Parameters:
    prices: dataframe of adjusted closing price data
    symbols: symbols to reference
    window: time window for all indicators that depend on moving average

    Returns: dataframe of all indicators
    '''
    indicators = pd.DataFrame(index = prices.index)
    indicators["Price"] = prices
    indicators["SMA"], indicators["Price/SMA"] = get_simple_moving_average(prices)
    indicators["STD"] = get_standard_deviation(prices)
    indicators["Upper BB"], indicators["Lower BB"], indicators["Normal BB"] = get_bollinger_bands(prices)
    indicators["Momentum"] = get_momentum(prices)
    return indicators

def calculate_portfolio_metrics(port_val):
    '''
    Calculate portfolio metrics.  Ported from Optimize Something project.

    Parameters:
    port_val: Dataframe of portfolio values
    Returns:
    cr: Cumulative return
    adr: Average daily return
    sddr: Standard deviation of daily return
    sr: Sharpe ratio (assuming daily trading k = 252)
    '''
    cr = port_val.iloc[-1,0]/port_val.iloc[0,0] - 1  # cumulative return
    dr = (port_val / port_val.shift(1) - 1).iloc[1:,0] # daily return
    adr = dr.mean() # average daily return
    sddr = dr.std() # standard deviation of daily return
    sr = math.sqrt(252)*(adr / sddr)  #Assume rorfr is 0
    return cr, adr, sddr, sr

def main_indicators():
    '''
    Main script for plotting indicators for report.
    '''

    #Get in-sample adj. close prices betweeen the date range.
    start_date_in_sample = dt.datetime(2008, 1, 1)
    end_date_in_sample = dt.datetime(2009, 12, 31)
    symbols = ["JPM"]
    prices_in = get_prices(symbols, start_date_in_sample, end_date_in_sample);
    
    #Calculate indicators
    indicators = get_indicators(prices_in, symbols)

    #Plot Price SMA data
    indicators[["Price", "SMA"]].plot(color = ["b","r"],linewidth = 0.7)
    plt.title("JPM Normalized Price and Simple Moving Average (Window = 10)", fontsize = 10)
    plt.ylabel("Normalized Values", fontsize = 8)
    plt.xlabel("Date Range", fontsize = 8)
    plt.savefig('SMA_1.png')
    
    #Plot Price/SMA ration
    indicators[["Price/SMA"]].plot(color = "b", linewidth = 0.7)
    plt.axhline(y=1, color = "r", linewidth =0.5, linestyle="--")
    plt.axhline(y=1.06, color = "r", linewidth = 0.5, linestyle = "--")
    plt.axhline(y=0.94, color = "r", linewidth = 0.5, linestyle = "--")
    plt.title("JPM Normalized Price to Simple Moving Average Ratio (Window = 10)", fontsize = 10)
    plt.ylabel("Normalized Values", fontsize = 8)
    plt.xlabel("Date Range", fontsize = 8)
    plt.savefig('SMA_2.png')
    
    #Plot Bollinger Bands
    indicators[["Price","SMA"]].plot(color =['b','m'], linewidth = 0.7)
    plt.plot(indicators[["Upper BB", "Lower BB"]], color = 'k', linewidth = 0.3, linestyle = "--")
    plt.title("Bollinger Bands for JPM Normalized Price \n(Window = 10, Band Size = 2 Standard Deviations)", fontsize = 10)
    plt.ylabel("Normalized Values", fontsize = 8)
    plt.xlabel("Date Range", fontsize = 8)
    plt.fill_between(indicators[["SMA"]].index, indicators[["SMA"]].values[:,0], indicators[["Upper BB"]].values[:,0], color = "g", alpha = 0.1)
    plt.fill_between(indicators[["SMA"]].index, indicators[["Lower BB"]].values[:,0], indicators[["SMA"]].values[:,0], color = "r", alpha = 0.1)
    plt.savefig('BB_1.png')

    #Plot Bollinger Normalized Value
    indicators[["Normal BB"]].plot(color = "b", linewidth = 0.7)
    plt.title("Normalized Bollinger Ratio", fontsize = 10)
    plt.ylabel("Normalized Values", fontsize = 8)
    plt.xlabel("Date Range", fontsize = 8)
    plt.axhline(y=1, color = "r", linewidth = 0.5, linestyle = "--")
    plt.axhline(y=-1, color = "r", linewidth = 0.5, linestyle = "--")
    plt.savefig('BB_2.png')

    #Plot Momentum
    indicators[["Momentum"]].plot(color = "b", linewidth = 0.7)
    plt.title("Momentum of Normalized JPM Price", fontsize=10)
    plt.ylabel("Normalized Values", fontsize = 8)
    plt.xlabel("Date Range", fontsize = 8)
    plt.axhline(y=0.4, color = "r", linewidth =0.5, linestyle="--")
    plt.axhline(y=0.0, color = "r", linewidth =0.5, linestyle="--")
    plt.axhline(y=-0.4, color = "r", linewidth =0.5, linestyle="--")
    plt.savefig('Momentum.png')

if __name__ == "__main__":
    main_indicators();