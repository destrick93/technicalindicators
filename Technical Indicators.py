#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import talib as ta
import ta
import yfinance as yf
import time
import numpy as np


# In[2]:


import talib as ta


# In[3]:


from ta import momentum


# In[4]:


df=yf.download('icicibank.ns',period='1y',interval='1d')


# In[5]:


def MACD(df,x,y,z):
    df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(df.Close, fastperiod=x, slowperiod=y, signalperiod=z)
    df.dropna(inplace=True)
    df.loc[df.macd>df.macdsignal,'new']=1
    df.loc[df.macd<df.macdsignal,'new']=-1
    df['pos']=df.new+df.new.shift(1)
    return df


# In[6]:


def UltimateOscillator(df,x,y,z):
    df['real'] = ta.ULTOSC(df.High,df.Low, df.Close, timeperiod1=x, timeperiod2=y, timeperiod3=z)


# In[7]:


def OHLC(ticker,x,y):
    yf.download(ticker,period=x,interval=y)


# In[8]:


def AO(df):
    df['SMA34'] = ta.SMA((df.Low+df.High)/2, timeperiod=34)
    df['SMA5'] = ta.SMA((df.Low+df.High)/2, timeperiod=5)
    df['AO']=df.SMA5-df.SMA34
    df=df.round(2)
    df.dropna(inplace=True)
    return df


# In[9]:


def ADR(df,x):
    df['DayRange']=abs(df.High-df.Low)
    df['AverageDayRange']=df.DayRange.rolling(x).mean()
    df.dropna(inplace=True)
    df=df.round(2)
    return df


# In[10]:


def BalanceofPower(df):
    df['BOP']=ta.BOP(df.Open,df.High,df.Low,df.Close)
    df=df.round(2)
    return df


# In[11]:


def WilliamR(df,x):
    df['William%R']=ta.WILLR(df.High, df.Low,df.Close, timeperiod=x)
    df.dropna(inplace=True)
    df=df.round(2)
    return df


# In[12]:


def WilliamFractal(df):
    Low=df.Low.tolist()
    High=df.High.tolist()
    bull=[]
    bear=[]
    for i in range(2,len(df)-2):
        if (High[i]>High[i-1]) and (High[i]>High[i-2]) and (High[i]>High[i+1]) and (High[i]>High[i+2]):
            bull.append(1)
        else:
            bull.append(0)
        if (Low[i]<Low[i-1]) and (Low[i]<Low[i-2]) and (Low[i]<Low[i+1]) and (Low[i]<Low[i+2]):
            bear.append(1)
        else:
            bear.append(0)
    bull.insert(len(bear), np.nan)
    bull.insert(len(bear)+1, np.nan)
    bull.insert(0, np.nan)
    bull.insert(1, np.nan)
    bear.insert(len(bear)+1, np.nan)
    bear.insert(len(bear), np.nan)
    bear.insert(0, np.nan)
    bear.insert(1, np.nan)
    df['bull']=np.array(bull)
    df['bear']=np.array(bear)
    df.dropna(inplace=True)
    return df


# In[13]:


def Stoch(df):
    df['fastk'],df['fastd'] = ta.STOCHF(df.High, df.Low,df.Close, fastk_period=14, fastd_period=14, fastd_matype=0)
    df['fastd']=df.fastk.rolling(3).mean()    
    df=df.round(2)
    df.dropna(inplace=True)
    return df


# In[14]:


def RVI(df,x):
    open=df.Open.tolist()
    close=df.Close.tolist()
    low=df.Low.tolist()
    high=df.High.tolist()
    num=[]
    den=[]
    for i in range(3,len(df)):
        a=close[i]-open[i]
        b=close[i-1]-open[i-1]
        c=close[i-2]-open[i-2]
        d=close[i-3]-open[i-3]
        num.append(a+(2*b)+(2*c)+d)
        e=high[i]-low[i]
        f=high[i-1]-low[i-1]
        g=high[i-2]-low[i-2]
        h=high[i-3]-low[i-3]
        den.append(e+(2*f)+(2*g)+h)
    num.insert(0,np.nan)
    num.insert(1,np.nan)
    num.insert(2,np.nan)
    den.insert(0,np.nan)
    den.insert(1,np.nan)
    den.insert(2,np.nan)
    df['den']=np.array(den)
    df['num']=np.array(num)
    df['den']=ta.SMA(df.den, timeperiod=x)
    df['num']=ta.SMA(df.num, timeperiod=x)
    df['rvi']=df.num/df.den
    df=df.round(4)
    signal=[]
    rvi=df.rvi.tolist()
    for i in range(3,len(df)):
        a=rvi[i]
        b=rvi[i-1]
        c=rvi[i-2]
        d=rvi[i-3]
        signal.append((a+(2*b)+(2*c)+d)/6)
    signal.insert(0,np.nan)
    signal.insert(1,np.nan)
    signal.insert(2,np.nan)
    df['signal']=np.array(signal)
    df=df.round(4)
    return df


# In[15]:


def HMA(df,x):
    a=x/2
    b=np.sqrt(x)
    df['Raw']=ta.WMA(df.Close, timeperiod=a)*2-ta.WMA(df.Close, timeperiod=x)
    b=np.sqrt(x)
    df['HMA']=ta.WMA(df.Raw,timeperiod=b)
    df=df.round(2)
    return df


# In[16]:


def MI(df,x):
    df['old']=df.High-df.Low
    df['ema9']=ta.EMA(df.old, timeperiod=9)
    df['ema_9']=ta.EMA(df.ema9, timeperiod=9)
    df['new']=df['ema9']/df['ema_9']
    df['MI']=df.emo.rolling(x).sum()
    df.dropna(inplace=True)
    df=df.round(2)
    return df


# In[17]:


def BOP(df):
    df['BOP']=(df.Close-df.Open)/(df.High-df.Low)
    df=df.round(2)
    return df


# In[18]:


def BB(df,x,y,z):
    df['upperband'], df['middleband'],df['lowerband'] = ta.BBANDS(df.Close, timeperiod=x, nbdevup=y, nbdevdn=y, matype=z)
    df=df.round(2)
    return df


# In[19]:


def BBB(df,x,y,z):
    df['upperband'], df['middleband'],df['lowerband'] = ta.BBANDS(df.Close, timeperiod=x, nbdevup=y, nbdevdn=y, matype=z)
    df['BB%']=(df.Close-df.lowerband)/(df.upperband-df.lowerband)
    df=df.round(2)
    return df


# In[20]:


def BBWidth(df,x,y,z):
    df['upperband'], df['middleband'],df['lowerband'] = ta.BBANDS(df.Close, timeperiod=x, nbdevup=y, nbdevdn=y, matype=z)
    df['BBWidth']=(df.upperband-df.lowerband)/df.middleband
    df=df.round(2)
    return df


# In[21]:


def DEMA(df,x):
    df['DEMA'] = ta.DEMA(close, timeperiod=x)


# In[22]:


def ADOSC(df,x,y):
    df['ADOSC'] = ta.ADOSC(df.High, df.Low, df.Close, df.Volume, fastperiod=x, slowperiod=y)
    return df


# In[23]:


def CMO(df,x):
    df['new']=df.Close.shift(1)
    df.loc[df.new<df.Close,'old']=df.Close-df.new
    df.loc[df.new>df.Close,'old1']=abs(df.Close-df.new)
    df=df.fillna(0)
    df['hell']=(df.old.rolling(x).sum()-df.old1.rolling(x).sum())/(df.old.rolling(x).sum()+df.old1.rolling(x).sum())
    df['hell']=df.hell*100
    df=df.round(2)
    return df


# In[24]:


def Coppock(df,x,y,z):
    df['roc']=ta.ROC(df.Close, timeperiod=x)+ta.ROC(df.Close, timeperiod=y)
    df['coppock']=ta.WMA(df.roc,timeperiod=z)
    df=df.round(2)
    return df


# In[25]:


def CCI(df,x,y):
    df['CCI'] = ta.CCI(df.High, df.Low, df.Close, timeperiod=x)
    ta.SMA(df.CCI, timeperiod=y)
    df=df.round(2)
    return df


# In[26]:


def KST(df,x,y,z,l,m,n,o,p,q):
    df['roc1']=ta.ROC(df.Close, timeperiod=x)
    df['roc2']=ta.ROC(df.Close,timeperiod=y)
    df['roc3']=ta.ROC(df.Close,timeperiod=z)
    df['roc4']=ta.ROC(df.Close,timeperiod=l)
    df['rcma1']=ta.SMA(df.roc1,timeperiod=m)
    df['rcma2']=ta.SMA(df.roc2,timeperiod=n)
    df['rcma3']=ta.SMA(df.roc3,timeperiod=o)
    df['rcma4']=ta.SMA(df.roc4,timeperiod=p)
    df['kst']=df.rcma1+(df.rcma2*2)+(df.rcma3*3)+(df.rcma4*4)
    df['signal']=ta.SMA(df.kst,timeperiod=q)
    df=df.round(4)
    return df


# In[27]:


def KeltnerChannel(df):
    df['mid']=ta.EMA(df.Close,timeperiod=20)
    df['atr']=ta.ATR(df.High, df.Low, df.Close, timeperiod=10)
    df['up']=df.mid+2*df.atr
    df['low']=df.mid-2*df.atr
    df=df.round(2)
    return df


# In[28]:


def KAMA(df,x,y,z):
    from ta.momentum import KAMAIndicator
    hello=KAMAIndicator(df.Close,window = x, pow1= y, pow2= z)
    df['kama']=hello.kama()


# In[29]:


def PPO(df,x,y,z):
    from ta.momentum import PercentagePriceOscillator
    hello=PercentagePriceOscillator(df.Close,window_slow= x, window_fast= y,window_sign= z)
    df['ppo']=hello.ppo()
    df['signal']=hello.ppo_signal
    df['ppohist']=hello.ppo_hist


# In[30]:


def TSI(df,x,y,z):
    from ta.momentum import TSIIndicator
    hello=TSIIndicator(df.Close, window_slow= x,window_fast = y)
    df['tsi']=hello.tsi()
    df['Signal']=ta.EMA(df.tsi,timeperiod=z)
    return df


# In[31]:


def AccDX(df):
    from ta.volume import AccDistIndexIndicator
    hello=AccDistIndexIndicator(df.High,df.Low,df.Close,df.Volume)
    df['Accdx']=hello.acc_dist_index()
    return df.Accdx


# In[32]:


def NVI(df):
    from ta.volume import NegativeVolumeIndexIndicator
    df['nvi']=NegativeVolumeIndexIndicator(df.Close,df.Volume).negative_volume_index()
    return df.nvi


# In[33]:


def OBV(df):
    from ta.volume import OnBalanceVolumeIndicator
    df['obv']=OnBalanceVolumeIndicator(df.Close,df.Volume).on_balance_volume()
    return df.obv


# In[34]:


def vwap(df):
    from ta.volume import volume_weighted_average_price
    df['vwap']=volume_weighted_average_price(df.High,df.Low,df.Close,df.Volume,window=1)
    return df.vwap


# In[35]:


def ADX(df,x):
    from ta.trend import ADXIndicator
    df['adx']=ADXIndicator(df.High, df.Low, df.Close,window= x).adx()
    return df.adx


# In[36]:


def DPO(df,x):
    from ta.trend import DPOIndicator
    df['dpo']=DPOIndicator(df.Close,window=x).dpo()
    return df.dpo


# In[37]:


def MassIndex(df,x,y):
    from ta.trend import MassIndex
    df['mi']=MassIndex(df.High,df.Low,window_fast=x,window_slow=y).mass_index()
    return df.mi


# In[38]:


def SMA(df,x):
    df['SMA']=ta.SMA(df.Close,timeperiod=14)
    return df


# In[39]:


def EOM(df,x,y):
    df['highn']=df.High.shift()
    df['lown']=df.Low.shift()
    df['t']=(df.High+df.Low)/2
    df['u']=(df.highn+df.lown)/2
    df['distnace']=df.t-df.u
    df['den']=(df.Volume/x)/(df.High-df.Low)
    df['final']=df.distnace/df.den
    df['eom']=ta.SMA(df.final,timeperiod=x)
    df.loc[df.eom<0,'eomfinal']='-0'
    df.loc[df.eom>0,'eomfinal']='0'
    df.dropna(inplace=True)
    return df


# In[40]:


def EMA(df,x):
    df['ema']=ta.EMA(df.Close,timeperiod=x)
    return df

