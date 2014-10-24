import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import netCDF4
# import sys
# sys.path.append('../moj')
# import jata
import matplotlib.pyplot as plt
from matplotlib import path
from turtleModule import dist, mon_alpha2num
from watertempModule import *
def np_datetime(m):
    if type(m) is str:
        year = int(m[5:9])
        month = mon_alpha2num(m[2:5])
        day =  int(m[0:2])
        hour = int(m[10:12])
        minute = int(m[13:15])
        second = int(m[-2:])
        dt = datetime(year,month,day,hour=hour,minute=minute,second=second)
    else:
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
'''
ctddata = pd.read_csv('ctd_good.csv')
shallow = ctddata.ix[17]        # 17, 19, 22
deep = ctddata.ix[19]        # 18914

shallowtime = np_datetime(shallow['END_DATE'])
shallowtemp = [float(temp) for temp in shallow['TEMP_VALS'].split(',')]
modelobj = water_roms()
modelurl = modelobj.get_url(shallowtime, shallowtime+timedelta(hours=1))
depth = [int(dep) for dep in shallow['TEMP_DBAR'].split(',')]
modeltemp = []
for dep in depth:
    print dep
    modeltemp.append(modelobj.watertemp(shallow['LON'], shallow['LAT'], dep,
                                            shallowtime, modelurl))

deeptime = np_datetime(deep['END_DATE'])
deeptemp = [float(temp) for temp in deep['TEMP_VALS'].split(',')]
modelobj2 = water_roms()
modelurl2 = modelobj2.get_url(deeptime, deeptime+timedelta(hours=1))
depth2 = [int(dep) for dep in deep['TEMP_DBAR'].split(',')]
modeltemp2 = []
print 'start deep'
for dep2 in depth2:
    modeltemp2.append(modelobj2.watertemp(deep['LON'], deep['LAT'], dep2,
                                              deeptime, modelurl2))

fig = plt.figure()
ax = fig.add_subplot(121)
# ax = fig.add_subplot(111)
ax.plot(modeltemp, depth, 'bo-', label='model')
ax.plot(shallowtemp, depth, 'ro-', label='obs')
ax.set_xlim([10, 30])
ax.set_ylim([40, 0])
ax.set_xlabel('Temp', fontsize=20)
ax.set_ylabel('Depth', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('%.2f, %.2f, %s' % (shallow['LAT'], shallow['LON'], str(shallowtime)), fontsize=20)
ax.legend(loc='lower right')

ax2 = fig.add_subplot(122)
ax2.plot(modeltemp2, depth2, 'bo-', label='model')
ax2.plot(deeptemp, depth2, 'ro-', label='deep')
ax2.set_xlim([10, 30])
ax2.set_ylim([40, 0])
ax2.set_xlabel('Temp')
ax2.set_ylabel('Depth')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax2.set_title('%.2f, %.2f, %s' % (deep['LAT'], deep['LON'], str(deeptime)),fontsize=20)
ax2.legend(loc='lower right')
plt.show()
'''
id = [17, 19, 22, 26]           # turtle ID you want to plot

ctddata = pd.read_csv('ctd_good.csv')
fig = plt.figure()
j = 1                           # for subplot
for i in id:
    data = ctddata.ix[i]
    tm = np_datetime(data['END_DATE'])
    temp = [float(temp) for temp in data['TEMP_VALS'].split(',')]
    modelobj = water_roms()
    modelurl = modelobj.get_url(tm, tm+timedelta(hours=1))
    depth = [int(dep) for dep in data['TEMP_DBAR'].split(',')]
    modeltemp = []
    for dep in depth:
        print dep
        modeltemp.append(modelobj.watertemp(data['LON'], data['LAT'], dep,
                                            tm, modelurl))
    ax = fig.add_subplot(2,2,j)
    ax.plot(modeltemp, depth, 'bo--', label='model',linewidth=2)
    ax.plot(temp, depth, 'ro-', label='obs', linewidth=2)
    ax.set_xlim([10, 30])
    ax.set_ylim([40, 0])
    ax.set_xlabel('Temp', fontsize=15)
    ax.set_ylabel('Depth', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_title('%.2f, %.2f, %s' % (data['LAT'], data['LON'], str(tm)), fontsize=18)
    ax.legend(loc='lower right')
    j+=1
plt.show()
