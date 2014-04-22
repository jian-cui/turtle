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
    return dt
def angle_conversion(a):
    a = np.array(a)
    return a/180*np.pi
def dist(lon1, lat1, lon2, lat2):
    # calculate the distance of points
    R = 6371.004
    lon1, lat1 = angle_conversion(lon1), angle_conversion(lat1)
    lon2, lat2 = angle_conversion(lon2), angle_conversion(lat2)
    l = R*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2)+
                    np.sin(lat1)*np.sin(lat2))
ctd = pd.read_csv('ctd_conversion.csv', index_col=0)
ctdlat = ctd['LAT']
ctdlon = ctd['LON']
ctdtime = np_datetime(ctd['END_DATE'])
gps = pd.read_csv('gps_conversion.csv', index_col=0)
gpslat = gps['LAT']
gpslon = gps['LON']
gpstime = np_datetime(gps['D_DATE'])
lonsize = [np.min(ctdlon), np.max(ctdlon)]
latsize = [np.min(ctdlat), np.max(ctdlat)]

fig = plt.figure()
ax = fig.add_subplot(111)
draw_basemap(fig, ax, lonsize, latsize)
ax.plot(ctdlon, ctdlat, 'b.', label='CTD')
# ax.plot(gpslon, gpslat, 'r.', label='GPS')
ax.set_title('turtle position')
plt.legend()
# plt.show()
index = []
i = 0
for lat, lon, ctdtm in zip(ctdlat, ctdlon, ctdtime):
    l, = distance((lat,lon), (gpslat,gpslon))
    # l = dist(lon, lat, gpslon, gpslat)
    p = np.where(l<3)
    maxtime = ctdtm + timedelta(hours=3)
    mintime = ctdtm - timedelta(hours=3)
    if mintime<gpstime[p[0]]<maxtime:
        index.append(i)
    i += 1
    print i
ctd_TF = pd.Series([True]*len(index), index=index)
ctd['TF'] = ctd_TF
print ctd
'''
run_number = 0
i = 0                           # index of 'ctd_conversion.csv'
ctd_right = dict(index=[],TIME=[], lat=[], lon=[])
ctd_wrong = dict(index=[],TIME=[], lat=[], lon=[])
tf = []
for lat, lon, ctdtm in zip(ctdlat, ctdlon, ctdtime):
    p = patches.Circle((lat, lon), radius=3)
    for la, lo, gpstm in zip(gpslat, gpslon, gpstime):
        run_number += 1
        print run_number
        if p.contains_point((lo, la)) and \
            ctdtm-timedelta(hours=2) < gpstm < ctdtm+timedelta(hours=2):
            ctd_right['lat'].append(lat)
            ctd_right['lon'].append(lon)
            ctd_right['TIME'].append(ctdtm)
            ctd_right['index'].append(i)
            tf.append('T')
        else:
            ctd_wrong['lat'].append(lat)
            ctd_wrong['lon'].append(lon)
            ctd_wrong['TIME'].append(ctdtm)
            ctd_wrong['index'].append(i)
            tf.append('F')
    i += 1
lost_percent = len(lat_wrong)/float(len(ctdlat))
print '{0} lost'.format(lost_percent)
ctd_r = pd.DataFrame(ctd_right)
ctd_w = pd.DataFrame(ctd_wrong)
ctd_r.to_csv('ctd_True.csv')
ctd_w.to_csv('ctd_False.csv')
tf_s = pd.Series(tf)
ctd['True'] = tf_s
ctd.to_csv('ctd_conversion_TF.csv')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
draw_basemap(fig2, ax2, lonsize, latsize)
ax.plot(ctd_right['lon'], ctd_right['lat'], 'b.', label="CTD right")
ax.plot(ctd_wrong['lon'], ctd_wrong['lat'], 'r.', label="CTD wrong")
plt.legend()
plt.show()
'''
