import pandas as pd
import patsy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


def dataAnalysis(data):
    '''
    This function takes in the dataset and provides
    plots to determine if the data is stationary process.
    '''
    names = data.columns
    time = data.pop(names[0])
    y = data.pop(names[1])
    x = data.pop(names[2])
    plt.subplot(311)
    tplot = time.diff().diff().plot()
    tplot.set_title('Delta '+names[0])
    plt.subplot(312)
    yplot = y.plot()
    yplot.set_title(names[1])
    yplot.set_xlabel(names[0])
    plt.subplot(313)
    xplot = x.plot()
    xplot.set_title(names[2])
    xplot.set_xlabel(names[0])
    plt.show()


def TFmodel(data,ARorder,diffOrder,exoOrder,exoDelay):
    '''
    This funciton provides a transfer function model.
    '''
    names = data.columns
    time = data.pop(names[0])
    yt = data.pop(names[1])
    xt = data.pop(names[2])
    data[names[0]] = time
    data[names[1]] = yt
    data[names[2]] = xt
    Xt = createExoMat(xt,exoOrder,names[2],time,exoDelay)
    yt.index = pd.to_datetime(time)
    yt = pd.TimeSeries(yt)
    yt = yt[(exoOrder+exoDelay):len(yt)]
    model = sm.tsa.ARIMA(yt, order = (ARorder,diffOrder,0), exog = Xt).fit(trend = 'nc')
    coefAnalysis(model)
    residPlots(model.resid,time[(exoOrder+exoDelay):len(time)])
    plot_fitted(data,model.params,ARorder,diffOrder,exoOrder,exoDelay)
    
    

def coefAnalysis(model):
    '''
    This function takes in a model and provides
    the parameters, their p-values and the
    standard error of the model.
    '''
    print "Coefficients"
    print model.params
    print " "
    print "P-Values"
    print model.pvalues
    print " "
    print "Mean Squared Error of Model"
    print np.sqrt(model.sigma2)
    print ' '


def residPlots(resids,Time):
    '''
    This function takes in the residuals of a model
    and returns plots to visualize the residuals of
    the model
    '''
    hist = plt.hist(resids)
    sm.graphics.qqplot(resids)
    sm.graphics.tsa.plot_pacf(resids, lags = 40)
    sm.graphics.tsa.plot_acf(resids, lags = 40)
    plt.show()


def createExoMat(Exo,q,name,Time,D):
    '''
    This helper function takes in a univariate exogeneous
    series and returns a lag of the coefficients

    Exo is the univaritate series, q is the the order of the
    lag terms, name is the name of the exogenous series, Time
    is the index of the exogenous series, and D is the delay
    in the exogenous series
    '''
    l = len(Exo)
    Time = Time[(q+D):l]
    X = pd.DataFrame(Exo[(D):l-q])
    if D != 0:
        x = X.pop(name)
        columnName = 'L'+str(D)+'.'+name
        x.columns = columnName
        X[columnName] = x
    X.index = pd.to_datetime(Time)
    for ii in range(q):
        columnName = 'L'+str(ii+1+D)+'.'+name
        tempSeries = pd.Series(Exo[D+ii+1:l-(q-ii-1)])
        tempSeries.name = columnName
        tempSeries.columns = columnName
        tempSeries.index = pd.to_datetime(Time)
        X[columnName] = tempSeries
    return X
        

def plot_fitted(data,params,endoOrder,diffOrder,exoOrder,delayOrder):
    '''
    This function plots both the actual value of the univariate
    time series yt and the predicted univariate time series yp.

    This function takes in the dataset (data) and the parameters
    from a model fit to the data (params), t
    '''
    names = data.columns
    time = data.pop(names[0])
    yt = data.pop(names[1])
    xt = data.pop(names[2])
    ly = len(yt)
    dx = xt
    dy = yt
    ICx = []
    ICy = []
    if diffOrder > 0:        
        for ii in range(diffOrder):
            ICx.append(dx[ii])
            ICy.append(dy[ii])
            dx = dx.diff()
            dy = dy.diff()
    xtCoef = params[0:exoOrder+1].values
    arCoef = params[exoOrder+1:exoOrder+endoOrder+1].values
    yp = seriesARConvolve(dy,arCoef)+seriesConvolve(dx,xtCoef)
    columnName = 'Predicted'+str(names[1])
    yp = pd.Series(yp,index = yt.index)
    yp.columns = columnName
    iy = yp
    if diffOrder > 0:
        iy = dfIntegrate(yp,ICy)
    error = yt.values-iy.values
    print "Standard Error "
    print (sp.var(np.nan_to_num(iy.values-yt.values)))**(0.5)
    print ' '
    plt.plot(time.values,iy.values)
    plt.plot(time.values,yt.values)
    plt.show()
    plt.plot(time.values,(yt.values-iy.values))
    plt.show()
    
def dfIntegrate(df,IC):
    '''
    This function takes in a dataframe that has been differenced
    and well as the intitial conditions for the differenced dataset
    and returns the original dataset.

    It takes in the difference dataframe df and the initial conditions
    that were lost during the differencing IC. It returns the "integrated"
    dataframe dfResults
    '''
    order = len(IC)
    dfResults = df
    for ii in range(order):
        dfTemp = dfResults
        dfResults.ix[(order-ii-1)] = IC[order-ii-1]
        dfResults
        for jj in range(len(df)-(order-ii)):
            dfResults.ix[jj+(order-ii)] = \
                    dfResults[jj+(order-ii)-1]+dfTemp[jj+(order-ii)]
    return dfResults


def seriesConvolve(xt,coef):
    '''
    This funciton takes in to time series and computes the convolution between
    them.
    '''
    #Check for Nans
    numNan = 0
    #7 was choosen with the assumption that diffOrder <= 7
    xtNan = np.isnan(xt).values[0:7]
    for ii in range(7): 
        if xtNan[ii]:
            numNan = numNan+1
    zt = np.convolve(xt.values[numNan:len(xt)],coef)
    zt = zt[len(coef)-1:len(zt)] #Truncate convolution signal to match original
    zt = np.roll(zt,len(coef)) #Aliegn Signals
    #Add Nans again
    for ii in range(numNan):
        np.insert(zt,0,0)
    zt[0:numNan] = xt.values[0]
    #Convert back to series
    df = pd.Series(zt)
    df.index = xt.index[0:len(zt)]
    return df

def seriesARConvolve(yt,coef):
    '''
    This funciton takes in to time series and computes the convolution between
    them.
    '''
    #Check for Nans
    numNan = 0
    #7 was choosen with the assumption that diffOrder <= 7
    ytNan = np.isnan(yt).values[0:7]
    for ii in range(7):
        if ytNan[ii]:
            numNan = numNan+1
    zt = np.convolve(np.insert(yt.values[numNan:len(yt)],0,0),coef) #Add 0 for AR process
    zt = zt[len(coef):len(zt)] #Truncate convolution signal to match original
    zt = np.roll(zt,len(coef)+numNan) #Aliegn Signals
    #Add Nans again
    for ii in range(numNan):
        np.insert(zt,0,0)
    zt[0:numNan] = yt.values[0]
    #Convert back to series
    df = pd.Series(zt)
    df.index = yt.index[0:len(zt)]
    return df

    
#To Do
#Add validation function
