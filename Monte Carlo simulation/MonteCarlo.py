import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr

def get_data(stocks, start, end):
    # old way of getting data
    # stockData = pd.concat(
    # {stock: pdr.DataReader(stock, "stooq")["Close"] for stock in stocks}, 
    # axis=1)
    # stockData = stockData.iloc[::-1]
    # stockData.index = pd.to_datetime(stockData.index)
    # stockData = stockData.loc[start:end]

    stockData = pdr.get_data_stooq(stocks, start, end)
    stockData = stockData["Close"]
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ["AAPL", "MSFT", "GOOGL"]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stockList, startDate, endDate)
print(meanReturns)

#give the stocks a random weight and normalise it to lie within [0,1]. 
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)
print(weights)

numSimulations = 100
timeframe = 100

meanM = np.full(shape=(timeframe, len(weights)), fill_value=meanReturns)
meanM = meanM.T
portfolioSimulations = np.full(shape=(timeframe, numSimulations), fill_value=0.0)
initialPortfolio = 10000
#MC loop
for m in range(0, numSimulations):
    #assume daily returns are distributed by a multivariate normal distribution
    #we will use cholesky decomposition to get lower triangular matrix
    Z = np.random.normal(size=(timeframe, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolioSimulations[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1) * initialPortfolio

plt.plot(portfolioSimulations)
plt.ylabel("Portfolio Value ($)")
plt.xlabel("Days")
plt.title("Monte Carlo simulation of a stock portfolio")
plt.show()
