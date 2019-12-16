import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import BagLearner as bag
import RandomTreeLearner as rt
import random
import TechnicalIndicators as ind

class StrategyLearner(object):

    def __init__(self, verbose = False, impact=0.0):
		'''
		Initializes the Strategy learner.
		'''
        self.verbose = verbose
        self.impact = impact
        self.window = 0
        self.learner = bag.BagLearner(rt.RandomTreeLearner, kwargs = {'leaf_size':5, 'verbose':verbose}, bags = 40, verbose=verbose)

    def get_indicators(self, prices, window = 10, band_size = 2):
        """
        Computes indicators for a symbol, given prices.

        Parameters:
        prices: array of prices
        window: time-window in days
        band_size: band_size for the Bollinger(R) Bands

        Returns:
        Tuple of indicators for price/SMA, momentum, BB
        """
        self.window = window
        sma, price_to_sma = ind.get_simple_moving_average(prices, window = window)
        momentum = ind.get_momentum(prices, window = window)
        upper, lower, bb_val = ind.get_bollinger_bands(prices, window = window, band_size = band_size)
        return price_to_sma, momentum, bb_val

    def indicators_to_features(self, indicators):
        """
        Converts indicators tuple into features list.

        Parameter: indicators - tuple of at least 3 indicators.
        Returns: single datafram of features from the indicators.
        """
        features = pd.concat([indicators[0], indicators[1]], axis=1)
        for i in range(2, len(indicators)):
            features = pd.concat([features, indicators[i]], axis=1)
        features.dropna(inplace=True)
        features.columns = ["feature{}".format(i) for i in range(len(features .columns))]
        return features 


    def addEvidence(self, symbol = "AAPL", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
		'''
		Trains learners on in-sample data set for trading.
		
		Parameters:
		Xtrain (nparray) - Observational data in feature columns
		Ytrain (nparray) - Ground-truth values for each observation
		'''
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates) # automatically adds SPY
        prices = prices_all[syms] # only portfolio symbols
        prices_SPY = prices_all['SPY'] # only SPY, for comparison later
        if self.verbose: print(prices)

        inds = self.get_indicators(prices, window = 10, band_size = 2)
        features = self.indicators_to_features(inds)
        features.drop(features.tail(1).index,inplace=True)
        dr = prices.pct_change().shift(periods = -1, axis = 0)
        x_train = features.values
        y_train = dr.loc[dr.index.isin(features.index)]
        y_train = y_train.values
        y_train[y_train > 1.02*self.impact] = 1
        y_train[y_train < 1.02*-self.impact] = -1
        y_train[(y_train< 1) & (y_train> -1)] = 0
        self.learner.addEvidence(x_train, y_train)
        
    def testPolicy(self, symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):
		'''
		Generates a dataframe of trades based on the learner's trading strategy policy.
		The policy is described in the attached report.
		
		Parameters:
		symbol (string) - The symbol with which to trade.
		sd (datetime) - Trading period start date.
		ed (datetime) - Trading period end date.
		sv (int) - Starting cash value.
		'''
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates) # automatically adds SPY
        prices = prices_all[syms] # only portfolio symbols
        prices_SPY = prices_all['SPY'] # only SPY, for comparison later
        if self.verbose: print(prices)

        # Get features from testing set
        inds = self.get_indicators(prices, window = 10, band_size = 2)
        features = self.indicators_to_features(inds)
        df_trades = pd.DataFrame(index = features.index);
        df_trades["Symbol"] = symbol
        df_trades["Order"] = "NONE"
        df_trades["Shares"] = 0
        features.drop(features.tail(1).index,inplace=True)
        x_test = features.values
        
        # Query and get Y-query
        y_test = self.learner.query(x_test)[0]
        y_test[y_test > self.impact] = 1
        y_test[y_test < -self.impact] = -1
        y_test[(y_test < 1) & (y_test > -1)] = 0
        y_test = list(y_test[0])
        y_test.append(0)

        holdings = 0
        position = 0
         
        for i in range(len(df_trades.index)):
            position = y_test[i]
            if position == -1:
                shares_to_trade = -1000-holdings
                holdings = -1000
            elif position == 1:
                shares_to_trade = 1000-holdings
                holdings = 1000
            elif position == 0:
                shares_to_trade = 0-holdings
                holdings = 0
            else:
                shares_to_trade = 0

            if shares_to_trade<0:
                df_trades.ix[i, "Order"] = "SELL"
                df_trades.ix[i, "Shares"]= -shares_to_trade
            elif shares_to_trade>0:
                df_trades.ix[i, "Order"] = "BUY"
                df_trades.ix[i, "Shares"] = shares_to_trade
        
        #df_trades = df_trades[df_trades.Order != "NONE"]
        df_trades2 = pd.DataFrame(index = prices.index.copy())
        df_trades2["Trade"] = 0
        for date in df_trades.index:
            if df_trades["Order"].loc[date] == "SELL":
                df_trades2["Trade"].loc[date] = df_trades["Shares"].loc[date]*-1
            if df_trades["Order"].loc[date] == "BUY":
                df_trades2["Trade"].loc[date] = df_trades["Shares"].loc[date]
        return df_trades2


if __name__=="__main__":
    slearner = StrategyLearner(impact = 0.0)
    slearner.addEvidence(symbol = 'JPM')
    df_trades = slearner.testPolicy(symbol = 'JPM')