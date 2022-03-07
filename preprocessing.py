import pandas as pd
import numpy as np
import os

"""

XLE = Energy
XLF = Financial Services
XLRE = Real Estate
XLP = Consumer Staples
XLB = Basic Materials
XLK = Technology
XLV = Healthcare
XLI = Industrial
XLY = Consumer Discretionary
XLU = Utilities
XLC = Communication Services

"""

def importData():
    outputFrame = pd.read_csv('Eth_2021.csv')[['Date', 'Change %']]
    outputFrame.columns = ['Date', 'ETH']
    # iterate over files in
    # that directory
    for filename in os.listdir('Data'):
        indicator = filename.split('_')[0]
        tmpFrame = pd.read_csv(f'Data/{filename}')[['Date', 'Change %']]
        tmpFrame.columns = ['Date', indicator]
        # want to inner merge b/c markets are closed on weekends and some holidays
        outputFrame = outputFrame.merge(tmpFrame, how='inner', on='Date')
    
    outputFrame = outputFrame.set_index('Date')

    for col in outputFrame.columns:
        outputFrame[col] = outputFrame[col].apply(lambda val: float(val[:-1])/100)

    return outputFrame


# print(importData())