'''
Extract data file ctd_extract_good.csv, add new column "TF".
If TF==True, data is good.
If TF==False, data is bad.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import patches
from datetime import datetime, timedelta
import random
import sys
sys.path.append('../moj')
from conversions import distance
def draw_basemap(fig, ax, lonsize, latsize, interval_lon=0.5, interval_lat=0.5):
    '''
    Draw basemap
    '''
    ax = fig.sca(ax)
    dmap = Basemap(projection='cyl',
                   llcrnrlat=min(latsize)-0.01,
                   urcrnrlat=max(latsize)+0.01,
                   llcrnrlon=min(lonsize)-0.01,
                   urcrnrlon=max(lonsize)+0.01,
                   resolution='h',ax=ax)
    dmap.drawparallels(np.arange(int(min(latsize)),
                                 int(max(latsize))+1,interval_lat),
                       labels=[1,0,0,0])
    dmap.drawmeridians(np.arange(int(min(lonsize))-1,
                                 int(max(lonsize))+1,interval_lon),
                       labels=[0,0,0,1])
    dmap.drawcoastlines()
    dmap.fillcontinents(color='grey')
    dmap.drawmapboundary()
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
    '''
    Return np.datetime.datetime from string
    '''
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
def angle_conversion(a):
    a = np.array(a)
    return a/180*np.pi
def dist(lon1, lat1, lon2, lat2):
    '''
    calculate the distance of points
    '''
    R = 6371.004
    lon1, lat1 = angle_conversion(lon1), angle_conversion(lat1)
    lon2, lat2 = angle_conversion(lon2), angle_conversion(lat2)
    l = R*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2)+
                    np.sin(lat1)*np.sin(lat2))
    return l
#####################################MAIN CODE##########################################
r = 3
hour = 3
ctd = pd.read_csv('2014_04_16_rawctd.csv')
ctdlat = ctd['LAT']
ctdlon = ctd['LON']
ctdtime = np_datetime(ctd['END_DATE'])
gps = pd.read_csv('2014_04_16_rawgps.csv')
gpslat = gps['LAT']
gpslon = gps['LON']
gpstime = np_datetime(gps['D_DATE'])
lonsize = [np.min(ctdlon), np.max(ctdlon)]
latsize = [np.min(ctdlat), np.max(ctdlat)]
'''
fig = plt.figure()
ax = fig.add_subplot(111)
draw_basemap(fig, ax, lonsize, latsize)
ax.plot(ctdlon, ctdlat, 'b.', label='CTD')
# ax.plot(gpslon, gpslat, 'r.', label='GPS')
ax.set_title('turtle position')
plt.legend()
plt.show()
'''
index = []
i = 0
for lat, lon, ctdtm in zip(ctdlat, ctdlon, ctdtime):
    # l, = distance((lat,lon), (gpslat,gpslon))
    l = dist(lon, lat, gpslon, gpslat)
    p = np.where(l<r)
    maxtime = ctdtm+timedelta(hours=hour)
    mintime = ctdtm-timedelta(hours=hour)
    mx = gpstime[p[0]]<maxtime
    mn = gpstime[p[0]]>mintime
    TF = mx*mn
    if TF.any():
        index.append(i)
    i += 1
    print i
ctd_TF = pd.Series([True]*len(index), index=index)
ctd['TF'] = ctd_TF
print ctd
print '{0} is OK(including "null" lon and lat values.).'.format(len(ctd_TF)/28975.0)
print '{0} is OK.'.format(len(ctd_TF)/15657.0)
print("save as 'ctd_extract_good.csv'")
ctd.to_csv('ctd_extract_good.csv')
