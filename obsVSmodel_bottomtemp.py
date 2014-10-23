'''
#draw the correlation of the deepest observation(we assume it's the bottom of sea) and appropriate model data.
'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import netCDF4
import matplotlib.pyplot as plt
from matplotlib import path
from scipy import stats
from watertempModule import *
from turtleModule import mon_alpha2num, np_datetime, bottom_value
FONTSIZE = 25
obs = pd.read_csv('ctd_good.csv')
tf_index = np.where(obs['TF'].notnull())[0] #indices of True data.
obsLon, obsLat = obs['LON'][tf_index], obs['LAT'][tf_index]
obsTime = pd.Series(np_datetime(obs['END_DATE'][tf_index]), index=tf_index)
obsTemp = pd.Series(bottom_value(obs['TEMP_VALS'][tf_index]), index=tf_index)
obsDepth = obs['MAX_DBAR'][tf_index]

starttime = datetime(2009,8,24)
endtime = datetime(2013,12,13)
tempobj = water_roms()
url = tempobj.get_url(starttime, endtime)
temp = tempobj.watertemp(obsLon.values, obsLat.values, obsDepth.values, obsTime.values, url)
temp = pd.Series(temp, index=tf_index)
i = temp[temp.isnull()==False].index

indexDeep = obs['MAX_DBAR'][i][obs['MAX_DBAR'][i]>50].index
indexShallow = obs['MAX_DBAR'][i][obs['MAX_DBAR'][i]<=50].index

tempModelDeep = temp[indexDeep]
tempModelShallow = temp[indexShallow]
tempObsDeep = obsTemp[indexDeep]
tempObsShallow = obsTemp[indexShallow]

x = np.arange(0.0, 30.0, 0.01)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(tempObsDeep, tempModelDeep, s=50, c='b')
ax1.plot(x, x, 'r-')
fit1 = np.polyfit(tempObsDeep, tempModelDeep, 1)
fit_fn1 = np.poly1d(fit1)
ax1.plot(tempObsDeep, fit_fn1(tempObsDeep), 'y-')
gradient1, intercept1, r_value1, p_value1, std_err1\
    = stats.linregress(tempObsDeep, tempModelDeep)
ax1.set_title('Deepest bottom & >50m, R-squard: %.4f' % r_value1**2, fontsize=FONTSIZE)
ax1.set_ylabel('Model temp', fontsize=FONTSIZE)
ax1.set_xlabel('OBS temp', fontsize=FONTSIZE)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(tempObsShallow, tempModelShallow, s=50, c='b')
ax2.plot(x, x, 'r-')
fit2 = np.polyfit(tempObsShallow, tempModelShallow, 1)
fit_fn2 = np.poly1d(fit2)
ax2.plot(tempObsShallow, fit_fn2(tempObsShallow), 'y-')
gradient2, intercept2, r_value2, p_value2, std_err2\
    = stats.linregress(tempObsShallow, tempModelShallow)
ax2.set_title('Deepest bottom & <50m, R-squard: %.4f' % r_value2**2, fontsize=FONTSIZE)
ax2.set_ylabel('Model temp', fontsize=FONTSIZE)
ax2.set_xlabel('OBS temp', fontsize=FONTSIZE)
plt.show()
