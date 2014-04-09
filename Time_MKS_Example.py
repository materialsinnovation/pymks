import pandas as pd
import numpy as np
import scipy as sp
from TransferFunction import dataAnalysis
from TransferFunction import TFmodel
from TransferFunction import coefAnalysis
from TransferFunction import residPlots
from TransferFunction import dfIntegrate
from prewhitening import xtModel
from prewhitening import prewhitening

fileLocation = 'C:\\Users\\PC\\Desktop\\DDHO.2.csv'
data = pd.io.parsers.read_csv(fileLocation)
#print type(data)

#names = data.columns
#data.names = names

#dataAnalysis(data)
#data = pd.io.parsers.read_csv(fileLocation)
TFmodel(data,4,0,0,0)


#Best fit for (2,0,1,0)
#xt roll = len(coef) yt roll = len(coef)-2
#Best fit for (2,1,1,0)
#xt roll = len(coef)-1 yt roll = len(coef)-1
#Best fit for (2,2,1,0)
#xt roll = len(coef)-2 yt roll = len(coef)-1
