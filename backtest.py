import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

class Backtest:
    def __init__(self, data: pd.DataFrame, predict: pd.DataFrame,  yMean: float, yStd: float, positionLimit: int = 1) -> None:
        """
        This class is used to backtest the model.
        """
        self.data = data
        self.principle = data.loc[0, "LastPrice"].iloc[0] * 5 # take the principle as 1*first price
        self.predict = predict
        self.mean = yMean
        self.std = yStd
        self.position = 0
        self.positionLimit = positionLimit
        self.PnLList = [] # the list to store the profit in every data item
        self.avgPrice = 0 # the average price of the position holding
        self.win = 0 # the number of rounds to get profit
        self.lose = 0 # the number of rounds to loss money
        self.askPrice = None
        self.askVolume = None
        self.bidPrice = None
        self.bidVolume = None

    def ask(self, price: int, volume: int = 1):
        """
        Here is the func to send ask order to the exchange.
        """
        if (self.position == self.positionLimit):
            return
        if (self.bidPrice is not None):
            self.cancel("bid")
        self.askPrice = price
        self.askVolume = volume
    
    def bid(self, price: int, volume: int = 1):
        """
        Here is the func to send bid order to the exchange.
        """
        if (self.position == -self.positionLimit):
            return
        if (self.askPrice is not None):
            self.cancel("ask")
        self.bidPrice = price
        self.bidVolume = volume
    
    def cancel(self, orderType: str):
        """
        Here is the func to cancel order.
        """
        if (orderType == "ask"):
            self.askPrice = None
            self.askVolume = None
        elif (orderType == "bid"):
            self.bidPrice = None
            self.bidVolume = None

    def update(self, ask1Price: int, bid1Price: int, LastPrice: int):
        """
        Here is the func to update my order and position.
        """
        self.PnLList.append(0)

        if (self.askPrice is not None):
            if (bid1Price <= self.askPrice):
                if (self.position == -1 and self.avgPrice > bid1Price):
                    self.win += 1
                if (self.position == -1 and self.avgPrice < bid1Price):
                    self.lose += 1
                self.PnLList[-1] += (bid1Price - self.avgPrice) * self.position * 5
                self.position += self.askVolume
                # commission charges
                self.PnLList[-1] -= 0.6 * self.askVolume
                self.askPrice = None
                if self.position == 0:
                    self.avgPrice = 0
                else:
                    self.avgPrice = bid1Price
                self.askVolume = None

        if (self.bidPrice is not None):
            if (ask1Price >= self.bidPrice):
                if (self.position == 1 and self.avgPrice < ask1Price):
                    self.win += 1
                if (self.position == 1 and self.avgPrice > ask1Price):
                    self.lose += 1
                self.PnLList[-1] += (ask1Price - self.avgPrice) * self.position * 5
                self.position -= self.bidVolume
                # commission charges
                self.PnLList[-1] -= 0.6 * self.bidVolume
                self.bidPrice = None
                if self.position == 0:
                    self.avgPrice = 0
                else:
                    self.avgPrice = ask1Price
                self.bidVolume = None
    def clean(self):
        """
        This func is used to clean the position and order.
        """
        if self.position > 0:
            self.cancel("bid")
            self.bid(0)
        elif self.position < 0:
            self.cancel("ask")
            self.ask(0)

    def cal_cumulative_return(self) -> list:
        """
        This func is used to calculate the cumulative return of trading.
        """
        cumulation = [0]
        for i in self.PnLList:
            cumulation.append(cumulation[-1] + i)
        return cumulation

    def cal_win_rate(self):
        """
        This func is used to calculate the winning rate of trading in a day.
        """
        return self.win/(self.win + self.lose)
    
    def cal_return_annually(self):
        """
        This func is used to calculate the annually return based on the return of five day.
        """
        return ((self.principle + np.sum(self.PnLList))/self.principle)**(365/5) - 1
    
    def cal_sharp_ratio(self):
        """
        This func is used to calculate the sharp ratio of the trading.
        """
        # As this startegy is based on the 1 min data, so we need to calculate the sharp ratio based on the 1 min return
        minReturn = np.array(self.PnLList[0:len(self.PnLList)//4*4:4]) + np.array(self.PnLList[1:len(self.PnLList)//4*4:4]) + np.array(self.PnLList[2:len(self.PnLList)//4*4:4]) + np.array(self.PnLList[3:len(self.PnLList)//4*4:4])
        return np.mean(minReturn)/np.std(minReturn)
    
    def cal_profit_withdraw_ratio(self):
        """
        This func is used to calculate the profit withdraw ratio of the trading.
        """
        cumulation = self.cal_cumulative_return()
        # use the accumulate max to minus return to get the max drawdown
        # add 1e-8 to avoid the divide by zero
        i = np.argmax((np.maximum.accumulate(cumulation)- np.array(cumulation)))
        if i == 0:
            return 0
        j = np.argmax(cumulation[:i])
        return cumulation[-1]*cumulation[j]/((cumulation[j] - cumulation[i])*self.principle + 1e-8)

    def run(self):
        """
        This func is used to run the backtest.
        """
        for i in range(len(self.data)):
            self.update(self.data.iloc[i]["AskPrice1"], self.data.iloc[i]["BidPrice1"], self.data.iloc[i]["LastPrice"])
            if self.data.index[i] >= 50000:
                # almost the end of the day, clean all the position
                self.clean()
                continue
            if self.data.iloc[i]["Time"] in self.predict.index:
                preReturn = self.predict.loc[self.data.iloc[i]["Time"]][0]
                print("get pre_return: ", preReturn)
            else:
                print("no pre_return")
                continue

            if (preReturn > self.mean):
                volume = 1
                if (preReturn > self.mean + 2*self.std and self.position == -1):
                    volume = 2
                if self.askPrice is not None:
                    self.cancel("ask")
                self.ask(self.data.iloc[i]["AskPrice1"], volume)
            elif (preReturn < self.mean):
                volume = 1
                if (preReturn < self.mean + 2*self.std  and self.position == 1):
                    volume = 2
                if self.bidPrice is not None:
                    self.cancel("bid")
                self.bid(self.data.iloc[i]["BidPrice1"], volume)