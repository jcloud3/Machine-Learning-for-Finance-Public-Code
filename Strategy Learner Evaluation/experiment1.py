import pandas as pd
import numpy as np
import random
import datetime as dt
import matplotlib.pyplot as plt
import MarketSimulator as ms
from util import get_data
import TechnicalIndicators as id
import StrategyLearner as sl
import ManualTradingStrategy as man
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def run_experiment_1(symbol, start_train, end_train, start_test, end_test, sv = 100000, commission = 0.0, impact = 0.0, output= "default.png"):

    np.random.seed(1000)
    random.seed(1000)

    slearner = sl.StrategyLearner(impact = impact)
    slearner.addEvidence(symbol = symbol, sd = start_train, ed = end_train, sv = sv)

    # In-Sample Experiment 1
    # Calculate manual strategy portfolio values
    manual_trades = man.testPolicy(symbol = symbol, sd = start_test, ed = end_test, sv = sv)
    manual_port_vals = ms.compute_portvals(manual_trades, start_val = sv, commission = commission, impact = impact)
    manual_port_vals = manual_port_vals / manual_port_vals.iloc[0,:]
    manual_port_vals.rename(columns = {"Portfolio Value":"Manual"}, inplace = True)

    # Calculate benchmark portfolio values
    dates = pd.date_range(start_test, end_test)
    syms = [symbol]
    price_range = get_data(syms, dates) # automatically adds SPY
    benchmark_trades = pd.DataFrame(data = [[symbol,"BUY",1000]], index = [price_range.index[0], price_range.index[-1]], columns = ["Symbol", "Order", "Shares"])
    bench_port_vals = ms.compute_portvals(benchmark_trades, start_val = sv, commission = commission, impact = impact)
    bench_port_vals = bench_port_vals / bench_port_vals.iloc[0,:]
    bench_port_vals.rename(columns = {"Portfolio Value":"Benchmark"}, inplace = True)

    # Calculate strategy learner portfolio values
    temp_strategylearner_trades = slearner.testPolicy(symbol = symbol, sd = start_test, ed = end_test, sv = sv)
    strategylearner_trades = pd.DataFrame(columns=['Order','Symbol','Shares'])  		   	  			  	 		  		  		    	 		 		   		 		  
    for row_idx in temp_strategylearner_trades.index:  		   	  			  	 		  		  		    	 		 		   		 		  
        nshares = temp_strategylearner_trades.loc[row_idx][0]  		   	  			  	 		  		  		    	 		 		   		 		  
        if nshares == 0:  		   	  			  	 		  		  		    	 		 		   		 		  
            continue  		   	  			  	 		  		  		    	 		 		   		 		  
        order = 'SELL' if nshares < 0 else 'BUY'  		   	  			  	 		  		  		    	 		 		   		 		  
        new_row = pd.DataFrame([[order,symbol,abs(nshares)],],columns=['Order','Symbol','Shares'],index=[row_idx,])  		   	  			  	 		  		  		    	 		 		   		 		  
        strategylearner_trades = strategylearner_trades.append(new_row)  		   	  			  	 		  		  		    	 		 		   		 		  

    strategylearner_port_vals = ms.compute_portvals(strategylearner_trades, start_val = sv, commission = commission, impact = impact)
    strategylearner_port_vals = strategylearner_port_vals / strategylearner_port_vals.iloc[0,:]
    strategylearner_port_vals.rename(columns = {"Portfolio Value":"Strategy"}, inplace = True)
    
    port_vals = pd.DataFrame(bench_port_vals["Benchmark"], index = bench_port_vals.index)
    port_vals["Manual"] = manual_port_vals["Manual"]
    port_vals["Strategy"] = strategylearner_port_vals["Strategy"]
    port_vals.fillna(method = 'ffill', inplace=True)


    # Plot in-sample strategy
    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.plot(port_vals["Benchmark"], color = 'g', linewidth = 0.7)
    plt.plot(port_vals["Manual"], color = 'r', linewidth = 0.7)
    plt.plot(port_vals["Strategy"], color = 'b', linewidth = 0.7)
    plt.legend(["Benchmark", "Manual", "StrategyLearner"])
    plt.title("Portfolio Values: Benchmark vs. Manual Strategy vs. StrategyLearner\n", fontsize = 10)
    plt.ylabel("Normalized Values", fontsize = 8)
    plt.xlabel("Date Range", fontsize = 8)
    plt.xticks(rotation = 45)
    plt.savefig(output)

    mcr, madr, msddr, msr = id.calculate_portfolio_metrics(manual_port_vals)
    bcr, badr, bsddr, bsr = id.calculate_portfolio_metrics(bench_port_vals)
    scr, sadr, ssddr, ssr = id.calculate_portfolio_metrics(strategylearner_port_vals)
    print("--------------------Manual Strategy--------------------")
    print("Cumulative Return:",  mcr)
    print("Average Daily Return:", madr)
    print("Stdev Daily Return:", msddr)
    print("Sharpe Ratio:", msr)
    print("--------------------Benchmark Strategy-----------------")
    print("Cumulative Return:",  bcr)
    print("Average Daily Return:", badr)
    print("Stdev Daily Return:", bsddr)
    print("Sharpe Ratio:", bsr)
    print("--------------------StrategyLearner---------------------")
    print("Cumulative Return:",  scr)
    print("Average Daily Return:", sadr)
    print("Stdev Daily Return:", ssddr)
    print("Sharpe Ratio:", ssr)
    
if __name__=="__main__":
    run_experiment_1("JPM", dt.datetime(2008,1,1), dt.datetime(2009,12,31), dt.datetime(2008,1,1), dt.datetime(2009,12,31), sv = 100000, commission = 0.0, impact = 0.000, output = "experiment1_in.png")
    run_experiment_1("JPM", dt.datetime(2008,1,1), dt.datetime(2009,12,31), dt.datetime(2010,1,1), dt.datetime(2011,12,31), sv = 100000, commission = 0.0, impact = 0.000, output = "experiment1_out.png")

