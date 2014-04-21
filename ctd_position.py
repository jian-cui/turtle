import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import patches
from datetime import datetime, timedelta
import random
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
i=0
lat_right, lon_right = [], []
lat_wrong, lon_wrong = [], []
for lat, lon, ctdtm in zip(ctdlat, ctdlon, ctdtime):
    p = patches.Circle((lat, lon), radius=3)
    for la, lo, gpstm in zip(gpslat, gpslon, gpstime):
        i += 1
        print i
        if p.contains_point((lo, la)) and \
           ctdtm-timedelta(hours=2) < gpstm < ctdtm+timedelta(hours=2):
            lat_right.append(lat)
            lon_right.append(lon)
        else:
            lat_wrong.append(lat)
            lon_wrong.append(lon)
lost_percent = len(lat_wrong)/float(len(ctdlat))
print '{0} lost'.format(lost_percent)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
draw_basemap(fig2, ax2, lonsize, latsize)
ax.plot(lon_right, lat_right, 'b.', label="CTD right")
ax.plot(lon_wrong, lat_wrong, 'r.', label="CTD wrong")
plt.legend()
plt.show()
