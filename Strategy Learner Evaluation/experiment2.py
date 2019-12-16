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

def run_experiment_2(symbol, start_train, end_train, start_test, end_test, sv = 100000, commission = 0.0, impact_states = (0.0, 0.015, 0.001), output= "default.png", verbose = False):
    impact_range = np.arange(impact_states[0], impact_states[1], impact_states[2])
    impact_range = np.round(impact_range, decimals = 3)
    df_exp2 = pd.DataFrame(columns = ['mcr', 'madr', 'msddr', 'msr', 'bcr', 'badr', 'bsddr', 'bsr', 'scr', 'sadr', 'ssddr', 'ssr'], index = list(impact_range))
    for column in df_exp2.columns:
        df_exp2[column] = 0
    for impact in impact_range:
        results = experiment_2_helper(symbol, start_train, end_train, start_test, end_test, sv=sv, commission=commission, impact = impact)
        df_exp2.loc[impact] = results
    if verbose:
        print(df_exp2)

    df_exp2['mcr'] /= df_exp2['mcr'].iloc[0]
    df_exp2['bcr'] /= df_exp2['bcr'].iloc[0]
    df_exp2['scr'] /= df_exp2['scr'].iloc[0]
    df_exp2['msddr'] /= df_exp2['msddr'].iloc[0]
    df_exp2['bsddr'] /= df_exp2['bsddr'].iloc[0]
    df_exp2['ssddr'] /= df_exp2['ssddr'].iloc[0]
    df_exp2['msr'] /= df_exp2['msr'].iloc[0]
    df_exp2['bsr'] /= df_exp2['bsr'].iloc[0]
    df_exp2['ssr'] /= df_exp2['ssr'].iloc[0]

    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.plot(df_exp2["bcr"], color = 'g', linewidth = 0.7)
    plt.plot(df_exp2["mcr"], color = 'r', linewidth = 0.7)
    plt.plot(df_exp2["scr"], color = 'b', linewidth = 0.7)
    plt.legend(["Benchmark", "Manual", "StrategyLearner"])
    plt.title("Cumulative Return vs Impact\n", fontsize = 10)
    plt.ylabel("Performance Metrics", fontsize = 8)
    plt.xlabel("Impact", fontsize = 8)
    plt.savefig("{}_cr.png".format(output))

    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.plot(df_exp2["badr"], color = 'g', linewidth = 0.7)
    plt.plot(df_exp2["madr"], color = 'r', linewidth = 0.7)
    plt.plot(df_exp2["sadr"], color = 'b', linewidth = 0.7)
    plt.legend(["Benchmark", "Manual", "StrategyLearner"])
    plt.title("Average Daily Return vs Impact\n", fontsize = 10)
    plt.ylabel("Performance Metrics", fontsize = 8)
    plt.xlabel("Impact", fontsize = 8)
    plt.savefig("{}_adr.png".format(output))

    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.plot(df_exp2["bsr"], color = 'g', linewidth = 0.7)
    plt.plot(df_exp2["msr"], color = 'r', linewidth = 0.7)
    plt.plot(df_exp2["ssr"], color = 'b', linewidth = 0.7)
    plt.legend(["Benchmark", "Manual", "StrategyLearner"])
    plt.title("Sharpe Ratio vs Impact\n", fontsize = 10)
    plt.ylabel("Performance Metrics", fontsize = 8)
    plt.xlabel("Impact", fontsize = 8)
    plt.savefig("{}_sr.png".format(output))

    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.plot(df_exp2["bsddr"], color = 'g', linewidth = 0.7)
    plt.plot(df_exp2["msddr"], color = 'r', linewidth = 0.7)
    plt.plot(df_exp2["ssddr"], color = 'b', linewidth = 0.7)
    plt.legend(["Benchmark", "Manual", "StrategyLearner"])
    plt.title("Standard Deviation of Daily Return vs Impact\n", fontsize = 10)
    plt.ylabel("Performance Metrics", fontsize = 8)
    plt.xlabel("Impact", fontsize = 8)
    plt.savefig("{}_sddr.png".format(output))

def experiment_2_helper(symbol, start_train, end_train, start_test, end_test, sv = 100000, commission = 0.0, impact = 0.0):

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

    mcr, madr, msddr, msr = id.calculate_portfolio_metrics(manual_port_vals)
    bcr, badr, bsddr, bsr = id.calculate_portfolio_metrics(bench_port_vals)
    scr, sadr, ssddr, ssr = id.calculate_portfolio_metrics(strategylearner_port_vals)
    return mcr, madr, msddr, msr, bcr, badr, bsddr, bsr, scr, sadr, ssddr, ssr
    
if __name__=="__main__":
    run_experiment_2("JPM", 
                     dt.datetime(2008,1,1), 
                     dt.datetime(2009,12,31), 
                     dt.datetime(2008,1,1), 
                     dt.datetime(2009,12,31), 
                     sv = 100000, 
                     commission = 0.0, 
                     output = "experiment2")

