import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from mpl_toolkits.basemap import Basemap
import pandas as pd
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

ctdo = pd.read_csv('2014_04_16_rawctd.csv')
ctdolat = ctdo['LAT']
ctdolon = ctdo['LON']
ctd = pd.read_csv('ctd_v2.csv', index_col=0)
ctdlat = ctd['LAT']
ctdlon = ctd['LON']
# gps = pd.read_csv('gps_conversion.csv')
gps = pd.read_csv('2014_04_16_rawgps.csv')
gpsv = gps['V_MASK']
gpslat = gps['LAT'][gpsv!=20]
gpslon = gps['LON'][gpsv!=20]
diag = pd.read_csv('2014_04_16_rawdiag.csv')
diaglat = diag['LAT']
diaglon = diag['LON']
# lonsize = [np.min(ctdlon), np.max(ctdlon)]
# latsize = [np.min(ctdlat), np.max(ctdlat)]

url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],u[0:1:69911][0:1:35][0:1:81][0:1:128],v[0:1:69911][0:1:35][0:1:80][0:1:129]'
data = netCDF4.Dataset(url)
lons, lats = data.variables['lon_rho'][:], data.variables['lat_rho'][:]
u, v = data.variables['u'][-1, 0, :, :], data.variables['v'][-1, 0, :, :]
lonsize = [np.amin(gpslon), np.amax(gpslon)]
latsize = [np.amin(gpslat), np.amax(gpslat)]

fig = plt.figure()
ax = fig.add_subplot(111)
draw_basemap(fig, ax, lonsize, latsize, interval_lon=1, interval_lat=1)
unlons = lons[u.mask]
unlats = lats[u.mask]
# plt.plot(unlons, unlats, 'b.')

romslons = lons[~u.mask]         # reverse the boolean value
romslats = lats[~u.mask]
# plt.plot(romslons, romslats, 'r.')
# plt.plot(ctdlon, ctdlat, 'y.', label="CTD")
plt.plot(gpslon, gpslat, 'g.', label="GPS")
# plt.plot(ctdolon, ctdolat, 'b.', label="CTD_original")
# fig.savefig('roms area', dpi=500)
plt.legend()
plt.show()
