import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from mpl_toolkits.basemap import Basemap
import pandas as pd
import sys
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
                       labels=[1,0,0,0], fontsize=15)
    dmap.drawmeridians(np.arange(int(min(lonsize))-1,
                                 int(max(lonsize))+1,interval_lon),
                       labels=[0,0,0,1], fontsize=15)
    dmap.drawcoastlines()
    dmap.fillcontinents(color='grey')
    dmap.drawmapboundary()

FONTSIZE = 25
while True:
    option = raw_input('Enter number to choose the figure yoy want to get:\n1:GoodCTD 2:RawCTD 3:GoodGPS 4:RawGPS 5:GoodDiag 6:RawDiag 0:quit\n')
    if option in ['1','2','3','4','5','6','0']: break
    print "Wrong Option, Please Choose Again"
fig =plt.figure()
ax = fig.add_subplot(111)
if option == '1':
    '''Don't need model area
    url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],u[0:1:69911][0:1:35][0:1:81][0:1:128],v[0:1:69911][0:1:35][0:1:80][0:1:129]'
    data = netCDF4.Dataset(url)
    lons, lats = data.variables['lon_rho'][:], data.variables['lat_rho'][:]
    u, v = data.variables['u'][-1, 0, :, :], data.variables['v'][-1, 0, :, :]
    lonsLand, latsLand = lons[u.mask], lats[u.mask]
    lonsOcean, latsOcean = lons[~u.mask], lats[~u.mask]   #reverse boolean value
    plt.plot(lonsLand, latsLand, 'b.', lonsOcean, latsOcean, 'r.', label='Model')
    '''
    ctd = pd.read_csv('ctd_extract_good.csv', index_col=0)
    TF = ctd['TF']
    latGoodCTD, lonGoodCTD = ctd['LAT'][TF==True], ctd['LON'][TF==True]
    plt.plot(lonGoodCTD, latGoodCTD, 'y.', label='GoodCTD')
    lonsize = [np.amin(lonGoodCTD), np.amax(lonGoodCTD)]
    latsize = [np.amin(latGoodCTD), np.amax(latGoodCTD)]
    plt.title('GoodCTD Positions', fontsize=FONTSIZE)
elif option == '2':
    ctd = pd.read_csv('2014_04_16_rawctd.csv')
    latRawCTD, lonRawCTD = ctd['LAT'], ctd['LON']
    lonsize = [np.amin(lonRawCTD), np.amax(lonRawCTD)]
    latsize = [np.amin(latRawCTD), np.amax(latRawCTD)]
    plt.plot(lonRawCTD, latRawCTD, 'b.', label='RawCTD')
    plt.title('RawCTD Positons', fontsize=FONTSIZE)
elif option == '3':
    gps = pd.read_csv('2014_04_16_rawgps.csv')
    lonGoodGPS, latGoodGPS = gps['LON'], gps['LAT']
    gpsv = gps['V_MASK']
    lonGoodGPS, latGoodGPS = lonGoodGPS[gpsv!=20], latGoodGPS[gpsv!=20]
    lonsize = [np.amin(lonGoodGPS), np.amax(lonGoodGPS)]
    latsize = [np.min(latGoodGPS), np.amax(latGoodGPS)]
    plt.plot(lonGoodGPS, latGoodGPS, 'g.', label='GoodGPS')
    plt.title('GoodGPS Positions', fontsize=FONTSIZE)
elif option == '4':
    gps = pd.read_csv('2014_04_16_rawgps.csv')
    lonRawGPS, latRawGPS = gps['LON'], gps['LAT']
    lonsize = [np.amin(lonRawGPS), np.amax(lonRawGPS)]
    latsize = [np.amin(latRawGPS), np.amax(latRawGPS)]
    plt.plot(lonRawGPS, latRawGPS, 'b.', label='RawGPS')
    plt.title('RawGPS Positons', fontsize=FONTSIZE)
elif option == '5':
    diag = pd.read_csv('2014_04_16_rawdiag.csv')
    diagV = diag['V_MASK']
    lonGoodDiag, latGoodDiag = diag['LON'][diagV!=20], diag['LAT'][diagV!=20]
    # lonsize = [-81, -62]
    # latsize = [30, 42]
    lonsize = [np.amin(lonGoodDiag), -25]
    latsize = [np.amin(latGoodDiag), np.amax(latGoodDiag)]
    plt.plot(lonGoodDiag, latGoodDiag, 'y.', label='GoodDiag')
    plt.title('GoodDiag Positions Within the Mid-Atlantic Region', fontsize=FONTSIZE)
elif option == '6':
    diag = pd.read_csv('2014_04_16_rawdiag.csv')
    lonRawDiag, latRawDiag = diag['LON'], diag['LAT']
    lonsize = [np.amin(lonRawDiag), np.amax(lonRawDiag)]
    latsize = [np.amin(latRawDiag), np.amax(latRawDiag)]
    plt.plot(lonRawDiag, latRawDiag, 'ro', label='RawDiag')
    plt.title('RawDiag Positions', fontsize=FONTSIZE)
elif option=='0':
    sys.exit()
draw_basemap(fig, ax, lonsize, latsize, interval_lon=20, interval_lat=20)
plt.legend()
plt.show()
