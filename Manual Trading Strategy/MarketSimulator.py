import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
import os  		   	  			  	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def compute_portvals(orders, start_val = 1000000, commission=9.95, impact=0.005):  		   	  			  	 		  		  		    	 		 		   		 		  
    '''
    Computes portfolio values for a given orders file with a starting portfolio value.
    Default commission rates of $9.95 and impact of 0.005 of transaction value.
    '''
    # Process order file
    orders, start_date, end_date, symbols, prices = process_orders(orders)
    # Process trades
    trades = process_trades(orders, prices, commission, impact)
    # Process shares
    shares = process_shares(orders, prices, trades, start_val)
    # Calculate portfolio values over time
    return calculate_portfolio(prices, shares)

def process_shares(orders, prices, trades, start_val):
    '''
    Helper function for processing shares based on trades.
    '''
    shares = pd.DataFrame(np.zeros((prices.shape)), index = prices.index, columns = prices.columns)
    shares.iloc[0,:-1] = trades.iloc[0,:-1].copy()
    shares.iloc[0,-1] = trades.iloc[0,-1] + start_val
    for row in range(1, len(shares)):
        shares.iloc[row] = shares.iloc[row-1] + trades.iloc[row]
    return shares

def process_trades(orders, prices, commission, impact):
    '''
    Helper function for processing all listed trades. Calculates for each trade in the order book:
      1. Change in asset values as a result of the trade.
      2. Transaction cost as a result of the trade.
      3. Modifies trades dataframe as a result of orders.
    '''
    trades = pd.DataFrame(np.zeros((prices.shape)), index = prices.index, columns = prices.columns)
    for index, row in orders.iterrows():
        trade_delta = prices.loc[index, row["Symbol"]] * row["Shares"]
        transaction_cost = impact*trade_delta + commission
        buy_or_sell = 1 if row["Order"]=="BUY" else -1
        trades.loc[index, row["Symbol"]] += buy_or_sell*row["Shares"]
        trades.loc[index, "Cash"] += -buy_or_sell*trade_delta - transaction_cost
    return trades

def calculate_portfolio(prices, shares):
    '''
    Helper function for calculating portfolio values given prices for all traded symbols
    and number of shares over time.
    '''
    share_value = prices*shares;
    portvals = pd.DataFrame(share_value.sum(axis = 1), share_value.index, ["Portfolio Value"])
    return portvals  

def process_orders(orders):
    '''
    Helper function for processing the given orders file, with NaN replacement.
    Returns a dataframe with all of the orders, sorted by ascending date.
    Returns a start_date, end_date, and list of symbols traded.
    Returns adjusted close prices for each of the symbols through the time window.
    '''
    orders.sort_index(ascending = True, inplace=True);
    sd = orders.index[0]
    ed = orders.index[-1]
    sym = orders["Symbol"].unique().tolist()
    p_d = get_data(sym, pd.date_range(sd, ed), addSPY=True)
    del p_d["SPY"]
    p_d.fillna(method='ffill', inplace = True)
    p_d.fillna(method='bfill', inplace=True)
    p_d["Cash"] = 1.0
    return orders, sd, ed, sym, p_d

def test_code():  		   	  			  	 		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		   	  			  	 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		   	  			  	 		  		  		    	 		 		   		 		  
    # Define input parameters  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    of = "./orders/orders-01.csv"  		   	  			  	 		  		  		    	 		 		   		 		  
    sv = 1000000  		   	  			  	 		  		  		    	 		 		   		 		  

    # Process orders  		   	  			  	 		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file = of, start_val = sv)  		   	  			  	 		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		   	  			  	 		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]] # just get the first column  		   	  			  	 		  		  		    	 		 		   		 		  
    else:  		   	  			  	 		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		   	  			  	 		  		  		    	 		 		   		 		  

    # Get portfolio stats  		   	  			  	 		  		  		    	 		 		   		 		  
    # Here we just fake the data. you should use your code from previous assignments.  		   	  			  	 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008,1,1)  		   	  			  	 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2008,6,1)  		   	  			  	 		  		  		    	 		 		   		 		  
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]  		   	  			  	 		  		  		    	 		 		   		 		  
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]  		   	  			  	 		  		  		    	 		 		   		 		  

    # Compare portfolio against $SPX  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals[-1]}")  		   	  			  	 		  		  		    	 		 		   		 		  

if __name__ == "__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    test_code()  		   	  			  	 		  		  		    	 		 		   		 		  
