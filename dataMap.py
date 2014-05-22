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

option = raw_input('Enter number to choose the figure yoy want to get\n\
                    1:CTD 2:CTD original 3:GPS 4:diag\n')
fig =plt.figure()
ax = fig.add_subplot(111)
if option == '1':
    url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],u[0:1:69911][0:1:35][0:1:81][0:1:128],v[0:1:69911][0:1:35][0:1:80][0:1:129]'
    data = netCDF4.Dataset(url)
    lons, lats = data.variables['lon_rho'][:], data.variables['lat_rho'][:]
    u, v = data.variables['u'][-1, 0, :, :], data.variables['v'][-1, 0, :, :]
    lonsize = [np.amin(lons), np.amax(lons)]
    latsize = [np.amin(lats), np.amax(lats)]
    lonsLand, latsLand = lons[u.mask], lats[u.mask]
    lonsOcean, latsOcean = lons[~u.mask], lats[~u.mask]   #reverse boolean value
    ctd = pd.read_csv('ctd_v2.csv', index_col=0)
    latCTD, lonCTD = ctd['LAT'], ctd['LON']
    plt.plot(lonsLand, latsLand, 'b.', lonsOcean, latsOcean, 'r.', label='Model')
    plt.plot(lonCTD, latCTD, 'y.', label='CTD')
elif option == '2':
    ctd = pd.read_csv('2014_04_16_rawctd.csv')
    latOCTD, lonOCTD = ctd['LAT'], ctd['LON']
    lonsize = [np.amin(lonOCTD), np.amax(lonOCTD)]
    latsize = [np.amin(latOCTD), np.amax(latOCTD)]
    plt.plot(lonOCTD, latOCTD, 'b.', label='CTD original')
elif option == '3':
    gps = pd.read_csv('2014_04_16_rawgps.csv')
    lonGPS, latGPS = gps['LON'], gps['LAT']
    gpsv = gps['V_MASK']
    lonGPS, latGPS = lonGPS[gpsv!=20], latGPS[gpsv!=20]
    lonsize = [np.amin(lonGPS), np.amax(lonGPS)]
    latsize = [np.min(latGPS), np.amax(latGPS)]
    plt.plot(lonGPS, latGPS, 'g.', label='GPS')
elif option == '4':
    diag = pd.read_csv('2014_04_16_rawdiag.csv')
    lonDiag, latDiag = diag['LON'], diag['LAT']
    lonsize = [np.amin(lonDiag), np.amax(lonDiag)]
    latsize = [np.min(latDiag), np.amax(latDiag)]
    plt.plot(lonDiag, latDiag, 'y.', label='Diag')
draw_basemap(fig, ax, lonsize, latsize, interval_lon=1, interval_lat=1)
plt.legend()
plt.show()
