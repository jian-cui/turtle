'''
#draw the correlation of the deepest observation(we assume it's the bottom of sea) and appropriate model data.
'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import netCDF4
import matplotlib.pyplot as plt
import sys
sys.path.append('../moj')
import jata
import utilities
from matplotlib import path
from scipy import stats
from watertempModule import *
def angle_conversion(a):
    a = np.array(a)
    return a/180*np.pi
def dist(lon1, lat1, lon2, lat2):
    '''
    calculate distance between 2 points
    '''
    R = 6371.004
    lon1, lat1 = angle_conversion(lon1), angle_conversion(lat1)
    lon2, lat2 = angle_conversion(lon2), angle_conversion(lat2)
    l = R*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2)+\
                        np.sin(lat1)*np.sin(lat2))
    return l
def mon_alpha2num(m):
    '''
    Return num from name of month
    '''
    month = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    if m in month:
        n = month.index(m)
    else:
        raise Exception('Wrong month abbreviation')
    return n+1
def np_datetime(m):
    dt = []
    for i in m:
        year = int(i[5:9])
        month = mon_alpha2num(i[2:5])
        day =  int(i[0:2])
        hour = int(i[10:12])
        minute = int(i[13:15])
        second = int(i[-2:])
        temp = datetime(year,month,day,hour=hour,minute=minute,second=second)
        dt.append(temp)
    dt = np.array(dt)
    return dt
def mean_value(v):
    v_list = []
    for i in v:
        print i, type(i)
        l = i.split(',')
        val = [float(i) for i in l]
        v_mean = np.mean(val)
        v_list.append(v_mean)
    return v_list
def bottom_value(v):
    v_list = []
    for i in v:
        l = i.split(',')
        val = float(l[-1])
        v_list.append(val)
    v_list = np.array(v_list)
    return v_list
def index_lv(v, n):
    '''
    return a dict
    '''
    index = {}
    for i in range(n):
        index[i] = []
    minv = np.min(v)
    maxv = np.max(v)+0.1
    m = (maxv - minv)/float(n)
    '''
    for i in range(n):
        minvv = minv + i*m
        maxvv = minv + (i+1)*m
        j = 0
        for val in v:
            if val>=maxvv and val<minvv:
                index[i].append(j)
    index = np.array(index)
    return index
    '''
    for i in range(len(v)):
        j = int((v.values[i] - minv)/m) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        index[j].append(v.index[i])
    return index
def index_by_depth(v, depth):
    i = {}
    i[0] = v[v<depth].index
    i[1] = v[v>=depth].index
    return i
##########################MAIN CODE#############################################
FONTSIZE = 25
ctd = pd.read_csv('ctd_good.csv')
tf_index = np.where(ctd['TF'].notnull())[0] #indices of True data.
ctdLon, ctdLat = ctd['LON'][tf_index], ctd['LAT'][tf_index]
ctdTime = pd.Series(np_datetime(ctd['END_DATE'][tf_index]), index=tf_index)
ctdTemp = pd.Series(bottom_value(ctd['TEMP_VALS'][tf_index]), index=tf_index)
ctdDepth = ctd['MAX_DBAR'][tf_index]

starttime = datetime(2009,8,24)
endtime = datetime(2013,12,13)
tempobj = water_roms()
url = tempobj.get_url(starttime, endtime)
temp = tempobj.watertemp(ctdLon.values, ctdLat.values, ctdDepth.values, ctdTime.values, url)
temp = pd.Series(temp, index=tf_index)
i = temp[temp.isnull()==False].index

indexDeep = ctd['MAX_DBAR'][i][ctd['MAX_DBAR'][i]>50].index
indexShallow = ctd['MAX_DBAR'][i][ctd['MAX_DBAR'][i]<=50].index

tempModelDeep = temp[indexDeep]
tempModelShallow = temp[indexShallow]
tempCTDDeep = ctdTemp[indexDeep]
tempCTDShallow = ctdTemp[indexShallow]

x = np.arange(0.0, 30.0, 0.01)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(tempCTDDeep, tempModelDeep, s=50, c='b')
ax1.plot(x, x, 'r-')
fit1 = np.polyfit(tempCTDDeep, tempModelDeep, 1)
fit_fn1 = np.poly1d(fit1)
ax1.plot(tempCTDDeep, fit_fn1(tempCTDDeep), 'y-')
gradient1, intercept1, r_value1, p_value1, std_err1\
    = stats.linregress(tempCTDDeep, tempModelDeep)
ax1.set_title('Deepest bottom & >50m, R-squard: %.4f' % r_value1**2, fontsize=FONTSIZE)
ax1.set_ylabel('Model temp', fontsize=FONTSIZE)
ax1.set_xlabel('CTD temp', fontsize=FONTSIZE)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(tempCTDShallow, tempModelShallow, s=50, c='b')
ax2.plot(x, x, 'r-')
fit2 = np.polyfit(tempCTDShallow, tempModelShallow, 1)
fit_fn2 = np.poly1d(fit2)
ax2.plot(tempCTDShallow, fit_fn2(tempCTDShallow), 'y-')
gradient2, intercept2, r_value2, p_value2, std_err2\
    = stats.linregress(tempCTDShallow, tempModelShallow)
ax2.set_title('Deepest bottom & <50m, R-squard: %.4f' % r_value2**2, fontsize=FONTSIZE)
ax2.set_ylabel('Model temp', fontsize=FONTSIZE)
ax2.set_xlabel('CTD temp', fontsize=FONTSIZE)
plt.show()
