'''
If the deepest observation depth is >50m(or <50m, or all), draw the correlation of this observation and appriate model data.
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
def mon_alpha2num(m):
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
def index_by_depth(v, depth):
    i = {}
    i[0] = v[v<depth].index
    i[1] = v[v>=depth].index
    return i
def show2pic(x1, y1, fontsize):
    FONTSIZE = fontsize
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    x = np.arange(0.0, 30.0, 0.01)
    '''
    for i in range(10):
        # ax.plot(temp[index[i]], obstemp[index[i]], '.', color=colors[i], label='{0}'.format(i))
        ax.scatter(temp[index[i]], obsdata['temp'][index[i]], s=50, c=colors[i], label='{0}'.format(i))
    '''
    # ax.scatter(temp[index[0]], obsdata['temp'][index[0]], s=50, c='b', label='<45')
    # ax.scatter(temp[index[1]], obsdata['temp'][index[1]], s=50, c='r', label='>=45')
    ax1.scatter(x1, y1, s=50, c='b')
    ax1.plot(x, x, 'r-', linewidth=2)
    plt.axis([0, 30, 0, 30], fontsize=15)
    plt.xlabel('Model temp', fontsize=FONTSIZE)
    plt.ylabel('CTD temp', fontsize=FONTSIZE)
    i = x1[x1.isnull()==False].index
    fit = np.polyfit(x1[i], y1[i], 1)
    fit_fn = np.poly1d(fit)
    x2, y2 = x1[i], fit_fn(x1[i])
    plt.plot(x2, y2,'y-', linewidth=2)
    gradient, intercept, r_value, p_value, std_err = stats.linregress(y1[i], x1[i])
    r_squared = r_value**2
    # ax1.set_title('R-squard: %.4f' % r_squared, fontsize=FONTSIZE)
    
    fig2 = plt.figure()
    ax2 =  fig2.add_subplot(111)
    nbins = 200
    H, xedges, yedges = np.histogram2d(x1[i], y1[i], bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0, H)
    plt.pcolormesh(xedges, yedges, Hmasked)
    plt.xlabel('Model temp', fontsize=FONTSIZE)
    plt.ylabel('CTD temp', fontsize=FONTSIZE)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts', fontsize=FONTSIZE)
    cbar.ax.set_yticks(fontsize=20)
    plt.axis([0, 30, 0, 30])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(x, x, 'r-', linewidth=2)
    plt.plot(x2, y2, 'y-', linewidth=2)
    # plt.title('R-squard: %.4f' % r_squared, fontsize=FONTSIZE)
    return ax1, ax2, r_squared
#############################MAIN CODE###########################################
FONTSIZE = 25
# obs = pd.read_csv('ctd_extract_TF.csv')
obs = pd.read_csv('ctd_good.csv')
tf_index = np.where(obs['TF'].notnull())[0]
obslat, obslon = obs['LAT'][tf_index].values, obs['LON'][tf_index].values
obstime = np_datetime(obs['END_DATE'][tf_index])
# obsdepth = mean_value(obs['TEMP_DBAR'][tf_index])
obsdepth = obs['MAX_DBAR'][tf_index].values
# obstemp = mean_value(obs['TEMP_VALS'][tf_index])
obstemp = bottom_value(obs['TEMP_VALS'][tf_index])
obsdata = pd.DataFrame({'depth':obsdepth, 'temp':obstemp, 'lon':obslon,
                        'lat':obslat, 'time':obstime}).sort_index(by='depth')

starttime = datetime(2009, 8, 24)
endtime = datetime(2013,12 ,13)
tempobj = water_roms()
url = tempobj.get_url(starttime, endtime)
# temp = tempobj.watertemp(obslon, obslat, obsdepth, obstime, url)
temp = tempobj.watertemp(obsdata['lon'].values, obsdata['lat'].values,
                         obsdata['depth'].values, obsdata['time'].values, url)
temp = pd.Series(temp, index = obsdata['temp'].index)

index = index_by_depth(obsdata['depth'], 50)
# colors = utilities.uniquecolors(10)
tp='all'
if tp == 'all':
    x1, y1 = temp, obsdata['temp']
    ax1, ax2, r_squared = show2pic(x1, y1, FONTSIZE)
    ax1.set_title('R-squared: %.4f' % r_squared, fontsize=FONTSIZE)
    ax2.set_title('R-squared: %.4f' % r_squared, fontsize=FONTSIZE)
elif tp == '<50':
    x1, y1 = temp[index[0]], obsdata['temp'][index[0]]
    ax1, ax2, r_squared = show2pic(x1, y1, FONTSIZE)
    ax1.set_title('%s, R-squared: %.4f' % (tp, r_squared), fontsize=FONTSIZE)
    ax2.set_title('%s, R-squared: %.4f' % (tp, r_squared), fontsize=FONTSIZE)
elif tp == '>50':
    x1, y1 = temp[index[1]], obsdata['temp'][index[1]]
    ax1, ax2, r_squared = show2pic(x1, y1, FONTSIZE)
    ax1.set_title('%s, R-squared: %.4f' % (tp, r_squared), fontsize=FONTSIZE)
    ax2.set_title('%s, R-squared: %.4f' % (tp, r_squared), fontsize=FONTSIZE)
plt.show()
'''
# Plot Deepest Data Quantity
fig = plt.figure()
ax = fig.add_subplot(111)
y = obsdata['depth'].values
x = np.arange(1, np.amax(y)+1)
bar = np.array([0]*np.amax(y))
for i in y:
    if i in x:
        bar[i-1] = bar[i-1]+1
plt.barh(x, bar)
plt.ylim([250, 0])
plt.ylabel('depth', fontsize=25)
plt.xlabel('Quantity', fontsize=25)
plt.title('Deepest data histogram', fontsize=25)
plt.show()
'''
