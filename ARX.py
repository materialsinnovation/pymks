import pandas as pd
import patsy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

'''
This function is used to determine the order of and calibrate the model that
will be used for the time history influence coefficients in MKS.

It takes in a csv file, as well as estimates for the order of the autoregressive
(AR) model, the number of difference that will be used, the order of the model
for the forcing function (exo), and value of the delay for the forcing function.

It returns the coefficients for the AR and the forcing function values as well
as the p-values for the coefficients. 

'''
def ARX(fileLocation,ARorder,diffOrder,exoOrder,exoDelay):
    '''
    This is the main function
    '''
    data = pd.io.parsers.read_csv(fileLocation)
    names = data.columns
    Time = data.pop(names[0])
    y = data.pop(names[1])
    x = data.pop(names[2])
    dataAnalysis(Time,y,x,names)
    X = createExoMat(x,exoOrder,names[2],Time,exoDelay)
    y.index = pd.to_datetime(Time)
    y = pd.TimeSeries(y)
    y = y[(exoOrder+exoDelay):len(y)]
    model = sm.tsa.ARIMA(y, order = (ARorder,diffOrder,0), exog = X).fit()
    coefAnalysis(model)
    residPlots(model.resid)


    '''
    This helper function generates plots to visuallize the data and to ensure
    it was sampled at a regular interval. 
    '''
def dataAnalysis(Time,y,x,names):
    plt.subplot(311)
    tplot = Time.diff().diff().plot()
    tplot.set_title('Delta '+names[0])
    plt.subplot(312)
    yplot = y.plot()
    yplot.set_title(names[1])
    yplot.set_xlabel(names[0])
    plt.subplot(313)
    xplot = x.plot()
    xplot.set_title(names[2])
    xplot.set_xlabel(names[0])
    #nonzero = Time.diff().diff()[Time.diff().diff() >= 1e-10].index[1]
    #if nonzero != len(Time.diff().diff()):
    #    print 'The time step changes as index',nonzero,'. Would you like to truncate the series?'
    plt.show()

    '''
    This helper function generates a print out the model parameters, their
    p - values, and the mean square value of the model with the data.
    '''
def coefAnalysis(model):
    print "Coefficients"
    print model.params
    print " "
    print "P-Values"
    print model.pvalues
    print " "
    print "Mean Squared Error"
    print np.sqrt(model.sigma2)  

    '''
    This function generates plots of the residuals of the model.
    '''
def residPlots(resids):
    hist = plt.hist(resids)
    sm.graphics.qqplot(resids)
    sm.graphics.tsa.plot_pacf(resids, lags = 40)
    sm.graphics.tsa.plot_acf(resids, lags = 40)
    plt.show()

    '''
    This function takes in the forcing function and returns a dataframe with
    each column as the lag of the forcing function.
    '''
def createExoMat(Exo,q,name,Time,D):
    l = len(Exo)
    Time = Time[(q+D):l]
    X = pd.DataFrame(Exo[(q+D):l])
    X.index = pd.to_datetime(Time)
    for ii in range(q):
        columnName = 'L.'+str(ii+1+D)+' '+name
        tempSeries = pd.Series(Exo[(q+D)-ii-1:l-ii-1])
        tempSeries.name = columnName
        tempSeries.columns = columnName
        tempSeries.index = pd.to_datetime(Time)
        X[columnName] = tempSeries
    return X
        


floc = 'C:\\Users\\PC\\Desktop\\PMMAdatashort.csv'
ARX(floc,3,2,0,0)

#To Do
#Cross Correlation
#Validate function with Box-Jenkins Data
