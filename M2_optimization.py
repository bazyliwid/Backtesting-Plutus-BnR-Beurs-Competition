# Practical trading:
# Select the starting and ending point of the investment, every 7 days compute:
# 1) the total cumulative return, total risk-free cumulative and their difference
# 2) weekly standard deviation (by downward rescaling of the total one for both investment and the benchmark)
# 3) Update the M2 in an separate array [later, more advanced, to be visualized as an indicator on a daily basis in the HTML display]
# => grid search optimizer cares only about the ending value therefore
# we just need to bring it to the end independently on the starting and ending point

# Simplifications (to be relaxed)
# 1) Go only Long 
# 2) Buy only full shares not fractional 
# 3) No constant / trailing stop loss or take profit 

# Install the foreign libraries 
# !pip install backtesting
# !pip install pandas_ta
# !pip install yfinance
import backtesting
from backtesting import Backtest, Strategy
from backtesting.test import GOOG # replace dataset if needed 
from backtesting.lib import crossover
import pandas_ta as ta
import numpy as np
import sys

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns 
import yfinance as yf
import math 
import pandas as pd 

# helper methods: 
def download_ticker_data(ticker, period, interval,investment_start,investment_end):
    df = yf.download(
            tickers = ticker,
            period = period,
            interval = interval
          )
    # ensure that the index is in datetime
    df.index = pd.to_datetime(df.index)

    if isinstance(df.columns, pd.MultiIndex):
     df.columns = df.columns.get_level_values(0)
    
    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)

    df = df[(df["Date"] >= investment_start) & (df["Date"] <= investment_end)]

    df.index = pd.to_datetime(df['Date'])
    df.drop('Date', axis=1, inplace=True)

    # print(df.head(5))
    # print(df.tail(5))
    return df

def indicator(data):
    bbands = ta.bbands(close = data["Close"], std = 1)
    # Get rid of the last column (band_percent)
    return bbands.to_numpy().T[0:3]

def get_ecb_yields(investment_start,investment_end):

    # ECB API URL for 12-month German bond yield
    ECB_API_URL = "https://sdw-wsrest.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y"
    headers = {"Accept": "application/json"}
    response = requests.get(ECB_API_URL, headers=headers)

    if response.status_code == 200:
        data = response.json()
        series_data = data["dataSets"][0]["series"]["0:0:0:0:0:0:0"]["observations"]
        dates = data["structure"]["dimensions"]["observation"][0]["values"]

        bond_data = []
        for i, (key, value) in enumerate(series_data.items()):
            date = datetime.strptime(dates[int(key)]["id"], "%Y-%m-%d")
            yield_value = float(value[0]) / 100  # Convert percentage to decimal
            bond_data.append([date, yield_value])

        # filters the yields based on the period provided 
        df = pd.DataFrame(bond_data, columns=["Date", "Yield"])
        df.sort_values("Date", ascending=True, inplace=True)
        df = df[(df["Date"] >= investment_start) & (df["Date"] <= investment_end)]
        df.index = pd.to_datetime(df['Date'])
        df.drop('Date', axis=1, inplace=True)

        return df
    else:
        print("Error fetching data:", response.status_code)
        return None


def M2_computation(c,f,rc_free,rce_asset,stde_asset,stde_benchmark):
  m2 = rc_free + rce_asset * max(stde_benchmark / max(stde_asset,f),(1/c)*(-rce_asset/abs(rce_asset)))
  return m2

# determine the entry and exit point of the investment and timespan 
investment_start = pd.to_datetime('2024-01-01')
investment_end = pd.to_datetime('2024-12-30')
backtest_frequency = "1d"
backtest_range = "5y"

# formula fixed parameters
c = 3
f = 0.002

# dowload the risk-free , asset and benchmark data for the given time-span 
df_benchmark = download_ticker_data("IWDE.MI", backtest_range, backtest_frequency,investment_start,investment_end)
df_asset = download_ticker_data("AAPL", backtest_range, backtest_frequency,investment_start,investment_end)
df_free = get_ecb_yields(investment_start,investment_end)

missing_dates = df_asset.index.difference(df_benchmark.index)

# Print the result
if missing_dates.empty:
    print("All indices in df_asset exist in df_benchmark.")
else:
    print(f"There is {len(missing_dates)} missing dates in df_benchmark:")
    print(missing_dates)
# print(df_benchmark)
# print(df_asset)

class BBStrategy(Strategy):
    
    # Main performance metrics 
    M2_total = 0 

    # define inner class attributes 
    benchmark = df_benchmark 
    data_f = df_free

    # define daily (prices) and weekly (returns, cululative) aggregators (for the cumulative returns computation)
    prices = {
        'benchmark': [],
        'portfolio': [],
        'free': [],
        }
  
    cumulative = {
        'benchmark': [],
        'portfolio': [],
        'free': [],
        }

    # To update the effective amount invested each week 
    # total_invested_this_week = 0 

    M2_weekly = []

    n = 10 

    def init(self):
        # Note that this returns each column as a row
        # Then extends the length of each row on each iteration
        self.sma = self.I(lambda x: pd.Series(x).rolling(self.n).mean(), self.data.Close)

    def next(self):
        # do not trade in non-aligning dates
        if self.data.index[-1] in pd.to_datetime(['2024-04-01', '2024-05-01', '2024-08-15', '2024-12-24', '2024-12-26', '2024-12-31']):
            return

        # print("I run")
        if crossover(self.data.Close, self.sma):  # If price crosses above SMA
            self.buy()
        elif crossover(self.sma, self.data.Close):  # If price drops below SMA
            self.position.close()  # Exit position (sell)

        # Get the current date / position value from the backtesting data
        current_date = self.data.index[-1] # extracts in YYYY-MM-DD format
        position_value = self.equity 

        # Update the daily parameters at each call 
        self.prices['benchmark'].append(self.benchmark.loc[current_date,'Close'])
        self.prices['portfolio'].append(position_value)
        self.prices['free'].append(self.data_f.loc[current_date,'Yield'])

        # I call this method after the trade logic therefore i get cash post-trades on Friday 
        #  self.prices['cash'].append(self.cash) -> unnecessary

        # Update the weekly parameters at each weekly call 
        if current_date.weekday() == 4:

          # the case of not having a Friday-to_Friday window yet, if we do not invest self.total_imvested_this_week is simply equal to 0 so no harm done 
          if len(self.prices['portfolio']) <= 4:
              
              # Capture the cash inflow / outflow 
              # if cash inflow to position self.prices['cash'][-1] < self.prices['cash'][0] so we substract it from total position value preventing artifical inflation 

              # returns computation 
              free_cum = (self.prices['free'][-1] - self.prices['free'][0])/self.prices['free'][0]
              self.cumulative['free'].append(free_cum)
              self.cumulative['portfolio'].append(((self.prices['portfolio'][-1] - self.prices['portfolio'][0]) / self.prices['portfolio'][0])-free_cum)
              self.cumulative['benchmark'].append(((self.prices['benchmark'][-1] - self.prices['benchmark'][0])/self.prices['benchmark'][0])-free_cum)

              # weekly std computation -> skipped since at the first observation mean equal to the only observation -> 0 

              # compute the weekly M2:
              self.M2_weekly.append(M2_computation(c,f,self.cumulative['free'][-1],self.cumulative['portfolio'][-1],0,0))
              self.M2_total = self.M2_weekly[-1]
              
          # the case of having a Friday-to_Friday window, use the stacked returns logic 
          else: 
            
            # Capture the cash inflow / outflow over last week (5 trading days)
            # self.total_invested_this_week = self.prices['cash'][-1] - self.prices['cash'][-5] -> useless, equity captures both position and cash 

            # returns computation 
            free_cum = (1 + (self.prices['free'][-1] - self.prices['free'][-5])/self.prices['free'][-5])*(1 + self.cumulative['free'][-1]) - 1
            self.cumulative['free'].append(free_cum)
            benchmark_cum = (1 + (self.prices['benchmark'][-1] - self.prices['benchmark'][-5])/self.prices['benchmark'][-5])*(1 + self.cumulative['benchmark'][-1]) - 1
            self.cumulative['benchmark'].append(benchmark_cum - free_cum)
            portfolio_cum = (1 + (self.prices['portfolio'][-1] - self.prices['portfolio'][-5])/self.prices['portfolio'][-5])*(1 + self.cumulative['portfolio'][-1]) - 1
            self.cumulative['portfolio'].append(portfolio_cum - free_cum)

            # weekly std computation
            mean_return_benchmark = np.sum(self.cumulative['benchmark']) / len(self.cumulative['benchmark'])  # Mean return
            std_benchmark = math.sqrt(np.sum((np.array(self.cumulative['benchmark']) - mean_return_benchmark) ** 2) / (len(self.cumulative['benchmark']) - 1))  

            mean_return_portfolio = np.sum(self.cumulative['portfolio']) / len(self.cumulative['portfolio'])  # Mean return
            std_portfolio = math.sqrt(np.sum((np.array(self.cumulative['portfolio']) - mean_return_portfolio) ** 2) / (len(self.cumulative['portfolio']) - 1))

            # compute the weekly M2:
            self.M2_weekly.append(M2_computation(c,f,self.cumulative['free'][-1],self.cumulative['portfolio'][-1],std_portfolio,std_benchmark))
            self.M2_total = self.M2_weekly[-1]
            

    

#### RUNNING THE STRATEGY 
bt = Backtest(df_asset, BBStrategy, cash=10_000)
stats = bt.run()

# Add M2 on top of the stats 
print("M2 recorded weekly: " + str(bt._strategy.M2_weekly) + "  ||   Number of weeks traded: " + str(len(bt._strategy.M2_weekly)))
M2_total = bt._strategy.M2_weekly[-1]  
stats = pd.concat([pd.Series({'M2_total': M2_total}), stats])  

print("")
print("")
print("Statistics, final M2 recorded on top: ")
print("")

# Print modified stats
print(stats)
# bt.plot()


#### OPTIMIZING THE STRATEGY 

# # create the user-defined optimization function 
# def optim_func(series):
#     if series['# Trades'] < 10:
#         return -1
#     else:
#         return series["Equity Final [$]"] / series["Exposure Time [%]"]

# stats, heatmap  = bt.optimize(
#     upper_bound = range(55,85,5),
#     lower_bound = range(10,45,5),
#     rsi_window = range(10,30,2),
#     # we could hardcode the M2 BnR Beurs metrics 
#     maximize = optim_func,
#     # create constraint for the optimization grid 
#     constraint = lambda param: param.upper_bound > param.lower_bound,
#     max_tries = 100, # apply randomised grid search if lower than the number of parameters, lower overfitting
#     return_heatmap = True
# )

# print(heatmap)

# # prepare heatmap to be visualized 
# hm = heatmap.groupby(["lower_bound","upper_bound"]).mean().unstack()
# sns.heatmap(hm, cmap = "plasma")
# plt.show()


print("I'm running !")


