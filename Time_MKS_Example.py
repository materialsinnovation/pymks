import pandas as pd
import numpy as np
from TransferFunction import dataAnalysis
from TransferFunction import TFmodel
from TransferFunction import coefAnalysis
from TransferFunction import residPlots
from prewhitening import xtModel
from prewhitening import prewhitening

fileLocation = 'C:\\Users\\PC\\Desktop\\DDHO.1.csv'
data = pd.io.parsers.read_csv(fileLocation)
#print type(data)

#names = data.columns
#data.names = names

#dataAnalysis(data)
#data = pd.io.parsers.read_csv(fileLocation)
TFmodel(data,2,2,1,0)

#data = pd.io.parsets.read_csv(fileLocation)
#xtModel(data,2,2,0)

#data = pd.io.parsers.read_csv(fileLocation)
#prewhitening(data,2,2,0)
