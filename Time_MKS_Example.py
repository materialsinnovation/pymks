import pandas as pd
import numpy as np
import scipy as sp
import TransferFunction as tf


fileLocation = 'C:\\Users\\PC\\Desktop\\DDHO.2.csv'
#calibrationData = pd.io.parsers.read_csv(fileLocation)
#tf.dataAnalysis(calibrationData)
calibrationData = pd.io.parsers.read_csv(fileLocation)
AR = 4
D = 0
Exo = 0
Delay = 0

model = tf.TFmodel(calibrationData,AR,D,Exo,Delay)
calibrationData = pd.io.parsers.read_csv(fileLocation)

tf.TFmodelAnalysis(calibrationData,model,AR,D,Exo,Delay)
fileLoc1 = 'C:\\Users\\PC\\Desktop\\DDHO.3.csv'
fileLoc2 = 'C:\\Users\\PC\\Desktop\\DDHO.4.csv'
data1 = pd.io.parsers.read_csv(fileLoc1)
data2 = pd.io.parsers.read_csv(fileLoc2)
tf.plot_predict(data1,model.params,AR,D,Exo,Delay)
tf.plot_predict(data2,model.params,AR,D,Exo,Delay)
data2 = pd.io.parsers.read_csv(fileLoc2)
predict = tf.yt_predict(data2,model.params,AR,D,Exo,Delay)
print predict.values[100:150]
