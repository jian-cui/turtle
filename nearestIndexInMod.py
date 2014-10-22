'''
Extract new data file named "ctd_good.csv "with new column "modNearestIndex" and "modDepthLayer"
'''
import pandas as pd
import numpy as np
import watertempModule as wtm
import netCDF4
from datetime import datetime, timedelta

def nearest_point_index2(lon, lat, lons, lats):
    d = wtm.dist(lon, lat, lons ,lats)
    min_dist = np.min(d)
    index = np.where(d==min_dist)
    return index
def str2float(arg):
    ret = []
    for i in arg:
        i = i.split(',')
        b = np.array([])
        for j in i:
            j = float(j)
            b = np.append(b, j)
        ret.append(b)
    ret = np.array(ret)
    return ret
def pointLayer(obsDepth, h, s_rho):
    #Return which layer is a certian point is in.
    index = nearest_point_index2(lon, lat, lons, lats)
    depthLayers = h[index[0][0]][index[1][0]] * s_rho
    # layerDepth = [depthLayers[-layer+1], depthLayers[-layer]]
    l = 36 - np.argmin(abs(depthLayers + vDepth))
    return l
obsData = pd.read_csv('ctd_extract_good.csv', index_col=0)
obsLon = obsData['LON']
obsLat = obsData['LAT']

starttime = datetime(2013,07,10)
endtime = starttime + timedelta(hours=1)
tempObj = wtm.water_roms()
url = tempObj.get_url(starttime, endtime)
modData = netCDF4.Dataset(url)
modLons = modData.variables['lon_rho'][:]
modLats = modData.variables['lat_rho'][:]
s_rho = modData.variables['s_rho'][:]
h = modData.variables['h'][:]
indexNotNull = obsLon[obsLon.isnull()==False].index
loc = []
for i in indexNotNull:
    ind = []
    lon = obsData['LON'][i]
    lat = obsData['LAT'][i]
    index = nearest_point_index2(lon, lat, modLons, modLats)
    ind.append(index[0][0])
    ind.append(index[1][0])
    loc.append(ind)
    print i
loc = pd.Series(loc, index=indexNotNull)
obsData['modNearestIndex'] = loc #add loc to obsData in case want to save it.

obsDepth = pd.Series(str2float(obsData['TEMP_DBAR']), index=obsData.index)
layersAll = []
for i in indexNotNull:
    nearest_index = loc[i]
    layers = []
    depthLayers = h[nearest_index[0], nearest_index[1]] * s_rho
    for j in range(len(obsDepth[i])):
        # depthLayers = h[nearest_index[0], nearest_index[1]] * s_rho
        l = np.argmin(abs(depthLayers+obsDepth[i][j]))
        layers.append(l)
        print i, j, l
    layersAll.append(layers)
layersAll = pd.Series(layersAll, index=indexNotNull)
obsData['modDepthLayer'] = layersAll
obsData.to_csv('ctd_good.csv')
