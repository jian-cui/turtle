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

ctd = pd.read_csv('ctd_v2.csv', index_col=0)
ctdlat = ctd['LAT']
ctdlon = ctd['LON']
gps = pd.read_csv('gps_conversion.csv')
gpslat = gps['LAT']
gpslon = gps['LON']
# lonsize = [np.min(ctdlon), np.max(ctdlon)]
# latsize = [np.min(ctdlat), np.max(ctdlat)]

url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],u[0:1:69911][0:1:35][0:1:81][0:1:128],v[0:1:69911][0:1:35][0:1:80][0:1:129]'
data = netCDF4.Dataset(url)
lons, lats = data.variables['lon_rho'][:], data.variables['lat_rho'][:]
u, v = data.variables['u'][-1, 0, :, :], data.variables['v'][-1, 0, :, :]
lonsize = [np.amin(lons), np.amax(lons)]
latsize = [np.amin(lats), np.amax(lats)]

fig = plt.figure()
ax = fig.add_subplot(111)
draw_basemap(fig, ax, lonsize, latsize)
unlons = lons[u.mask]
unlats = lats[u.mask]
plt.plot(unlons, unlats, 'b.')

uselons = lons[~u.mask]         # reverse the boolean value
uselats = lats[~u.mask]
plt.plot(uselons, uselats, 'r.')
# plt.plot(ctdlon, ctdlat, 'y.', label="CTD")
plt.plot(gpslon, gpslat, 'g.', label="GPS")
# fig.savefig('roms area', dpi=500)
plt.legend()
plt.show()
