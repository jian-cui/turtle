'''
Draw error bar
'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from  watertempModule import np_datetime, bottom_value, dist
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import watertempModule as wtm         # A module of classes that using ROMS, FVCOM.
from turtleModule import str2ndlist, np_datetime, bottom_value, dist
def nearest_point_index2(lon, lat, lons, lats):
    '''
    Calculate the nearest point.
    '''
    d = wtm.dist(lon, lat, lons ,lats)
    min_dist = np.min(d)
    index = np.where(d==min_dist)
    return index
def pointLayer(lon, lat, lons, lats, vDepth, h, s_rho):
    '''
    Return which layer is a certian point is in.
    '''
    index = nearest_point_index2(lon, lat, lons, lats)
    depthLayers = h[index[0][0]][index[1][0]] * s_rho
    # layerDepth = [depthLayers[-layer+1], depthLayers[-layer]]
    l = np.argmin(abs(depthLayers + vDepth))
    return l
###################################MAIN CODE###########################################
FONTSIZE = 25
obs = pd.read_csv('ctd_extract_good.csv') # From ctd_extract_TF.py
tf_index = np.where(obs['TF'].notnull())[0]
obsLon, obsLat = obs['LON'][tf_index], obs['LAT'][tf_index]
obsTime = pd.Series(np_datetime(obs['END_DATE'][tf_index]), index=tf_index)
obsTemp = pd.Series(str2ndlist(obs['TEMP_VALS'][tf_index]), index=tf_index)
# obsTemp = pd.Series(bottom_value(obs['TEMP_VALS'][tf_index]), index=tf_index)
obsDepth = pd.Series(str2ndlist(obs['TEMP_DBAR'][tf_index]), index=tf_index)

starttime = datetime(2009, 8, 24)
endtime = datetime(2013, 12, 13)
tempObj = wtm.waterCTD()
url = tempObj.get_url(starttime, endtime)
modTemp = tempObj.watertemp(obsLon.values, obsLat.values, obsDepth.values, obsTime.values, url)

d = {'lon': obsLon, 'lat': obsLat, 'obstemp': obsTemp.values,
     'modtemp':modTemp, 'depth': obsDepth, 'time': obsTime.values}
a = pd.DataFrame(d, index=tf_index)

ind = [] # the indices needed
obst = []
modt = []
dep = []
for i in a.index:
    for j in range(len(a['obstemp'][i])):
        print i, j
        y = a['modtemp'][i][j]
        x = a['obstemp'][i][j]
        if y > x + 10:
            ind.append(i)
            obst.append(x)
            modt.append(y)
            dep.append(a['depth'][i][j])
dataFinal = pd.DataFrame({'lon': a['lon'][ind].values,
                          'lat': a['lat'][ind].values,
                          'time': a['time'][ind].values,
                          'obstemp': np.array(obst),
                          'modtemp': np.array(modt),
                          'dep': np.array(dep)
                          })
starttime = datetime(2013,07,10)
endtime = starttime + timedelta(hours=1)
layer = 4                       # the first layer you want to plot

tempObj = wtm.water_roms()
url = tempObj.get_url(starttime, endtime)
modData = tempObj.get_data(url)
h, s_rho = modData['h'], modData['s_rho']
lons, lats = modData['lon_rho'], modData['lat_rho']

lyrs = []
for i in dataFinal.index:
    l = pointLayer(dataFinal['lon'][i],dataFinal['lat'][i], lons, lats, dataFinal['dep'][i], h, s_rho)
    lyrs.append(l)
    print i, l
dataFinal['layer'] = pd.Series(lyrs, index=dataFinal.index)

lonsize = np.amin(lons)-0.1, np.amax(lons)+0.1
latsize = np.amin(lats)-0.1, np.amax(lats)+0.1

fig = plt.figure()
ax = []
i = 0
for i in range(0, 4):
    ax.append(plt.subplot(2,2,i+1))
    layer = layer+i*4
    lon, lat = dataFinal.ix[5]['lon'], dataFinal.ix[5]['lat']
    a = np.where(dataFinal['layer']==layer)[0]
    m = dataFinal['time'][a]>starttime-timedelta(days=10)
    n = dataFinal['time'][a]<starttime+timedelta(days=10)
    b = np.where(m & n)[0]
    indx = dataFinal.ix[a].index[b]
    colorValues = dataFinal['obstemp'][indx]/32
    modLayerTemp = tempObj.layerTemp(layer, url)  #grab new layer temp
    fig.sca(ax[i])
    dmap = Basemap(projection = 'cyl',
               llcrnrlat = min(latsize)-0.01,
               urcrnrlat = max(latsize)+0.01,
               llcrnrlon = min(lonsize)-0.01,
               urcrnrlon = max(lonsize)+0.01,
               resolution = 'h', ax=ax[i])
    dmap.drawparallels(np.arange(int(min(latsize)), int(max(latsize))+1, 2),
                   labels = [1,0,0,0])
    dmap.drawmeridians(np.arange(int(min(lonsize)), int(max(lonsize))+1, 2),
                   labels = [0,0,0,1])
    dmap.drawcoastlines()
    dmap.fillcontinents(color='grey')
    dmap.drawmapboundary()
    if i==0:
        c = plt.contourf(lons, lats, modLayerTemp, extend ='both')
        clrmap = c.cmap
    else:
        c = plt.contourf(lons, lats, modLayerTemp, extend ='both', cmap=clrmap)
    plt.scatter(dataFinal['lon'][indx], dataFinal['lat'][indx], s=40, c=colorValues.values, cmap=clrmap)
    ax[i].set_title('Layer: {0}'.format(layer))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
plt.colorbar(c, cax=cbar_ax, ticks=range(0, 32, 4))     #c is the contour of first subplot
plt.suptitle('obsVSmodel, modTemp>obsTemp+10',fontsize=25)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(1,37)
bar = np.array([0]*36)
for i in dataFinal['layer']:
    if i in x:
        print i
        bar[i-1] = bar[i-1]+1
plt.bar(x, bar)
plt.xlabel('Layer', fontsize=25)
plt.ylabel('Quantity', fontsize=25)
plt.show()

#draw errorbar based on depth.
fig = plt.figure()
ax = fig.add_subplot(111)
y = dataFinal['dep'].order().values
x = np.arange(1, np.amax(y)+1)
bar = np.array([0]*np.amax(y))
for i in y:
    if i in x:
        bar[i-1] = bar[i-1]+1
plt.barh(x, bar)
plt.ylim((50, 0))
plt.ylabel('depth', fontsize=25)
plt.xlabel('Quantity', fontsize=25)
plt.title('error bar, based on depth',fontsize=25)
plt.show()
