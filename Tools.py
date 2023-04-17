import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def changeFreq(data: pd.DataFrame, freq: str):
    """
    This func is used to change the data freq to get a DataFrameGroupBy object.
    """
    data.loc[:, "Time"] = data.loc[:, "date"] + data.loc[:, "Time"].apply(str)
    data.loc[:, "Time"] = pd.to_datetime(data.loc[:, 'Time'], format='%Y-%m-%d%H%M%S%f')
    return data.groupby(pd.Grouper(key="Time", freq=freq))

def lastPrice(data: pd.DataFrame) -> int:
    """
    This func is used to get the newest price in the bar.
    """
    if len(data.loc[:, "LastPrice"]) == 0:
        return np.nan
    return data.loc[:, "LastPrice"].iloc[-1]

def highPrice(data: pd.DataFrame) -> int:
    """
    This func is used to get the highest price in the bar.
    """
    if len(data.loc[:, "LastPrice"]) == 0:
        return np.nan
    return data.loc[:, "LastPrice"].max()

def lowPrice(data: pd.DataFrame) -> int:
    """
    This func is used to get the lowest price in the bar.
    """
    if len(data.loc[:, "LastPrice"]) == 0:
        return np.nan
    return data.loc[:, "LastPrice"].min()

def turnover(data: pd.DataFrame) -> int:
    """
    This func is used to get the turnover in the bar.
    """
    if len(data.loc[:, "Turnover"]) == 0:
        return np.nan
    return data.loc[:, "Turnover"].iloc[-1]

def volume(data: pd.DataFrame) -> int:
    """
    This func is used to get the volume in the bar.
    """
    if len(data.loc[:, "Volume"]) == 0:
        return np.nan
    return data.loc[:, "Volume"].iloc[-1]

def openInterest(data: pd.DataFrame) -> int:
    """
    This func is used to get the change of openinterest in the bar.
    """
    if len(data.loc[:, "OpenInterest"]) == 0:
        return np.nan
    return data.loc[:, "OpenInterest"].iloc[-1]

def maxBidAskRate(data: pd.DataFrame) -> int:
    """
    This func is used to get the max rate of bid ask volume weighted average price in the bar.
    the rate of bid ask weighted average price = bid1*volume/ask1*volume
    """
    if len(data) == 0:
        return np.nan
    return ((data.loc[:, "AskPrice1"] * data.loc[:, "AskVol1"]) / (data.loc[:, "BidPrice1"] * data.loc[:, "BidVol1"])).max()

def extractReturn(price: pd.Series) -> pd.Series:
    """
    This func is used to get the return in two bars.
    """
    returnData = price.diff()/price.shift(1)
    returnData.dropna(inplace = True)
    return returnData

def extractVolume(volume: pd.Series) -> pd.Series:
    """
    This func is used to get the volume in two bars.
    """
    returnData = volume.diff()
    returnData.dropna(inplace = True)
    return returnData

def extractTurnover(turnover: pd.Series) -> pd.Series:
    """
    This func is used to get the turnover in two bars.
    """
    returnData = turnover.diff()
    returnData.dropna(inplace = True)
    return returnData

def extractDeltaOpenInterest(openInterest: pd.Series) -> pd.Series:
    """
    This func is used to get the change of openinterest in two bars.
    """
    returnData = openInterest.diff()/openInterest
    returnData.dropna(inplace = True)
    return returnData