__author__ = 'teja'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
import scipy.stats
import statsmodels.api as sm
from ggplot import *



def main():
    #import data files.
    turnstile_weather = pd.read_csv('turnstile_data_master_with_weather.csv');
    turnstile_weather_regress = pd.read_csv('turnstile_weather_v2.csv');

    with_rain_frame = turnstile_weather.ix[turnstile_weather.rain == 1, 'ENTRIESn_hourly']
    without_rain_frame = turnstile_weather.ix[turnstile_weather.rain == 0, 'ENTRIESn_hourly']

    #Explore the distribution of the samples by plotting a histogram.
    p = plotHistogram(with_rain_frame,without_rain_frame)
    p.show() # distrbutions are not normal,need to use non-parametric tests.

    #Perform Mann-Whitney Test
    U, p = scipy.stats.mannwhitneyu(with_rain_frame, without_rain_frame, use_continuity=True)
    print "The U Statistic values is "+str(U)
    print "One tailed p-value is %f" %p

    #Two tailed p values = 0.0249*2 ie < 0.05. So there is a significant difference in the distributions of samples
    #Means and medians of sample can be get more insigts about the data
    with_rain_mean = np.mean(with_rain_frame)
    without_rain_mean = np.mean(without_rain_frame)
    with_rain_median = np.median(with_rain_frame)
    without_rain_median = np.median(without_rain_frame)
    print 'The mean of entries with rain is ' + str(with_rain_mean)
    print 'The mean of entries without rain is ' + str(without_rain_mean)
    print 'The median of entries with rain is ' + str(with_rain_median)
    print 'The median of entries without rain is ' + str(without_rain_median)
    #Mean and median of sample with rain is higher than the sample without rain.


    #imporoved data is used for regression. This dataset is read to a dataframe turnstile_weather_regress;
    predictions = regress(turnstile_weather_regress)
    #Plot the residuals
    residual_plot = plot_residuals(turnstile_weather_regress, predictions)
    residual_plot.show()
    # compute r-square for the model
    r_square = compute_r_squared(turnstile_weather_regress, predictions)
    print 'R-squared for the model is ' + str(r_square)

    #Visualizations
    hourly_plot,temp_pressure_plot = visualizations_with_ggplot(turnstile_weather)
    print hourly_plot,temp_pressure_plot

def linear_regression(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]
    return intercept, params

def regress(dataframe):
    features = dataframe[['rain', 'meantempi']]
    #dummy coding
    dummy_units = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    dummy_conds = pd.get_dummies(dataframe['conds'], prefix='conds')
    dummy_day_week = pd.get_dummies(dataframe['day_week'], prefix='day_week')
    dummy_hour = pd.get_dummies(dataframe['hour'], prefix='hour')
    features = features.join(dummy_units).join(dummy_conds).join(dummy_day_week).join(dummy_hour)
    # Values
    values = dataframe['ENTRIESn_hourly']
    # Perform linear regression
    intercept, params = linear_regression(features, values)
    print 'Params of rain is ' + str(params['rain'])
    print 'Params of meantempi is ' + str(params['meantempi'])
    #Predicted values form the regression equation.
    predictions = intercept + np.dot(features, params)
    return predictions

def plot_residuals(dataframe,predictions):
    fig = plt.figure()
    ax = fig.add_subplot(111);
    fig.suptitle('Histogram of Residuals', fontsize=14, fontweight='bold')
    ax.set_xlabel('Residual Error')
    ax.set_ylabel('Frequency')
    (dataframe['ENTRIESn_hourly'] - predictions).hist()
    return plt

def compute_r_squared(dataframe,predictions):
    mean = np.mean(dataframe['ENTRIESn_hourly'])
    SST = ((dataframe['ENTRIESn_hourly'] - mean) ** 2).sum()
    SSReg = ((predictions - dataframe['ENTRIESn_hourly']) ** 2).sum()
    r_square = 1 - (SSReg / SST)
    return r_square

def plotHistogram(with_rain_frame,without_rain_frame):
    fig = plt.figure()
    ax = fig.add_subplot(111);
    fig.suptitle('Histogram of Entriesn_hourly', fontsize=14, fontweight='bold')
    #Patches for adding to legend
    blue_patch = mpatches.Patch(color='blue', label='With Rain')
    green_patch = mpatches.Patch(color='green', label='Without Rain')
    plt.legend(handles=[blue_patch,green_patch])
    ax.set_xlabel('ENTRIESn_HOURLY')
    ax.set_ylabel('Frequency')
    without_rain_frame.hist(bins=100)
    with_rain_frame.hist(bins=100)
    ax.set_xlim([0,6000])
    return plt

def visualizations_with_ggplot(dataframe):
    hourly_data = dataframe.groupby('Hour',as_index = False).mean()
    hourly_plot = ggplot(hourly_data, aes(x='Hour',y='ENTRIESn_hourly')) + \
                  geom_line() +\
                  geom_point() +\
                  ggtitle('Entries based on hour') + xlab('Hour of the day') + ylab('Average entries')

    temp_pressure = dataframe.ix[:,['meanpressurei','meanwindspdi','meantempi']].drop_duplicates()
    temp_pressure_plot =  ggplot(temp_pressure, aes('meantempi','meanpressurei')) + \
    geom_point(colour='steelblue')+\
    ggtitle('Temperature and Pressure')+ xlab('Mean Temperature') + ylab('Mean Pressure')

    return hourly_plot,temp_pressure_plot;

if __name__ == "__main__":
    main()
