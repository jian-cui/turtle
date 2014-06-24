import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import watertempModule as wtm
from  watertempModule import np_datetime, bottom_value, dist
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from module import str2ndlist
import netCDF4
def str2float(arg):
    ret = []
    for i in arg:
        a = i.split(',')
        b = [float(j) for j in a]
        ret.append(b)
    ret = np.array(ret)
    return ret
def nearest_point_index2(lon, lat, lons, lats):
    d = wtm.dist(lon, lat, lons ,lats)
    min_dist = np.min(d)
    index = np.where(d==min_dist)
    return index
def pointLayer(lon, lat, lons, lats, vDepth, h, s_rho):
    #Return which layer is a certian point is in.
    index = nearest_point_index2(lon, lat, lons, lats)
    depthLayers = h[index[0][0]][index[1][0]] * s_rho
    # layerDepth = [depthLayers[-layer+1], depthLayers[-layer]]
    l = 36 - np.argmin(abs(depthLayers + vDepth))
    return l
def closest_num(num, numlist, i=0):
    '''
    Return index of the closest number in the list
    '''
    index1, index2 = 0, len(numlist)
    indx = int(index2/2)
    if not numlist[0] < num < numlist[-1]:
        raise Exception('{0} is not in {1}'.format(str(num), str(numlist)))
    if index2 == 2:
        l1, l2 = num-numlist[0], numlist[-1]-num
        if l1 < l2:
            i = i
        else:
            i = i+1
    elif num == numlist[indx]:
        i = i + indx
    elif num > numlist[indx]:
        i = closest_num(num, numlist[indx:],
                          i=i+indx)
    elif num < numlist[indx]:
        i = closest_num(num, numlist[0:indx+1], i=i)
    return i
def getModTemp(modTempAll, ctdTime, ctdLayer, ctdNearestIndex, starttime, oceantime):
    ind = closest_num((starttime -datetime(2006,01,01)).total_seconds(), oceantime)
    modTemp = []
    l = len(ctdLayer.index)
    for i in ctdLayer.index:
        print i, l
        timeIndex = closest_num((ctdTime[i]-datetime(2006,01,01)).total_seconds(), oceantime)-ind
        modTempTime = modTempAll[timeIndex]
        modTempTime[modTempTime.mask] = 10000
        t = [modTempTime[ctdLayer[i][j]-1,ctdNearestIndex[i][0], ctdNearestIndex[i][1]] \
             for j in range(len(ctdLayer[i]))]
        modTemp.append(t)
    modTemp = np.array(modTemp)
    return modTemp
FONTSIZE = 25
ctdData = pd.read_csv('ctd_good.csv')
tf_index = np.where(ctdData['TF'].notnull())[0]
ctdLon, ctdLat = ctdData['LON'][tf_index], ctdData['LAT'][tf_index]
ctdTime = pd.Series(np_datetime(ctdData['END_DATE'][tf_index]), index=tf_index)
ctdTemp = pd.Series(str2float(ctdData['TEMP_VALS'][tf_index]), index=tf_index)
# ctdTemp = pd.Series(bottom_value(ctd['TEMP_VALS'][tf_index]), index=tf_index)
ctdDepth = pd.Series(str2float(ctdData['TEMP_DBAR'][tf_index]), index=tf_index)
ctdLayer = pd.Series(str2ndlist(ctdData['modDepthLayer'][tf_index],bracket=True), index=tf_index)
ctdNearestIndex = pd.Series(str2ndlist(ctdData['modNearestIndex'][tf_index], bracket=True), index=tf_index)

starttime = datetime(2009, 8, 24)
endtime = datetime(2013, 12, 13)
tempObj = wtm.waterCTD()
url = tempObj.get_url(starttime, endtime)
# modTemp1 = tempObj.watertemp(ctdLon.values, ctdLat.values, ctdDepth.values, ctdTime.values, url)
modDataAll = tempObj.get_data(url)
oceantime = modDataAll['ocean_time']
modTempAll = modDataAll['temp']
modTemp = getModTemp(modTempAll, ctdTime, ctdLayer, ctdNearestIndex, starttime, oceantime)

d = {'lon': ctdLon, 'lat': ctdLat, 'obstemp': ctdTemp.values,
     'modtemp':modTemp, 'depth': ctdDepth, 'time': ctdTime.values,
     'layer': ctdLayer}
a = pd.DataFrame(d, index=tf_index)
'''
a = pd.read_csv('temp.csv',index_col=0)
modTime = []
for i in a['time']:
    modTime.append(datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))
modTime = pd.Series(modTime, index=a.index)
tDepth = []
for i in a['']
'''
ind = [] # the indices needed
obst = []
modt = []
lyr = []
dep=[]
for i in a.index:
    # obst = []
    # modt = []
    # dep = []
    for j in range(len(a['obstemp'][i])):
        print i, j
        y = a['modtemp'][i][j]
        x = a['obstemp'][i][j]
        # if abs(x - y) > 10:
        if abs(x - y) > 10:          # obstemp-modtemp>10
            ind.append(i)
            obst.append(x)
            modt.append(y)
            lyr.append(a['layer'][i][j])
            dep.append(a['depth'][i][j])
dataFinal = pd.DataFrame({'lon': a['lon'][ind].values,
                          'lat': a['lat'][ind].values,
                          'time': a['time'][ind].values,
                          'obstemp': np.array(obst),
                          'modtemp': np.array(modt),
                          'layer': np.array(lyr),
                          'dep': np.array(dep)
                          })
starttime = datetime(2013,07,10)
endtime = starttime + timedelta(hours=1)
layer = 4
'''
tempObj = wtm.water_roms()
url = tempObj.get_url(starttime, endtime)
lon, lat = dataFinal.ix[5]['lon'].values[0], dataFinal.ix[5]['lat'].values[0]
modTemp, layerDepth = tempObj.layerTemp(lon, lat, depth, url)
modData = tempObj.get_data(url)
lons, lats = modData['lon_rho'], modData['lat_rho']

a = abs(np.max(layerDepth))<dataFinal['dep']
b = dataFinal['dep']<abs(np.min(layerDepth))
i = np.where(a & b)[0]    #layerDepth is negative
c = dataFinal['time'][i]>starttime-timedelta(days=10)
d = dataFinal['time'][i]>starttime+timedelta(days=10)
j = np.where(c & d)[0]
indx = dataFinal.ix[i].index[j]
colorValues = dataFinal['obstemp'][indx]/32
lonsize = np.amin(lons)-0.1, np.amax(lons)+0.1
latsize = np.amin(lats)-0.1, np.amax(lats)+0.1
fig = plt.figure()
ax = plt.subplot(111)
dmap = Basemap(projection = 'cyl',
           llcrnrlat = min(latsize)-0.01,
           urcrnrlat = max(latsize)+0.01,
           llcrnrlon = min(lonsize)-0.01,
           urcrnrlon = max(lonsize)+0.01,
           resolution = 'h', ax=ax)
dmap.drawparallels(np.arange(int(min(latsize)), int(max(latsize))+1, 2),
               labels = [1,0,0,0])
dmap.drawmeridians(np.arange(int(min(lonsize)), int(max(lonsize))+1, 2),
               labels = [0,0,0,1])
dmap.drawcoastlines()
dmap.fillcontinents(color='grey')
dmap.drawmapboundary()
c = plt.contourf(lons, lats, modTemp, extend='both')
plt.colorbar()
plt.scatter(dataFinal['lon'][indx], dataFinal['lat'][indx], s=40,c = colorValues.values, cmap=c.cmap)
plt.show()
'''
tempObj = wtm.water_roms()
url = tempObj.get_url(starttime, endtime)

modData = tempObj.get_data(url)
h, s_rho = modData['h'], modData['s_rho']
lons, lats = modData['lon_rho'], modData['lat_rho']
'''
#get the mod layer. Unusefule after using "ctd_good.csv".
lyrs = []
for i in dataFinal.index:
    l = pointLayer(dataFinal['lon'][i],dataFinal['lat'][i], lons, lats, dataFinal['dep'][i], h, s_rho)
    lyrs.append(l)
    print i, l
dataFinal['layer'] = pd.Series(lyrs, index=dataFinal.index)
'''
lonsize = np.amin(lons)-0.1, np.amax(lons)+0.1
latsize = np.amin(lats)-0.1, np.amax(lats)+0.1

fig = plt.figure()
ax = []
# clrmap = mpl.colors.Colormap('s', 9)
i = 0
ax.append(plt.subplot(2,2,i+1))
modLayerTemp = tempObj.layerTemp(layer, url)
l = layer+i*4
# lon, lat = dataFinal.ix[5]['lon'], dataFinal.ix[5]['lat']
a = np.where(dataFinal['layer']==l)[0]
m = dataFinal['time'][a]>starttime-timedelta(days=10)
n = dataFinal['time'][a]<starttime+timedelta(days=10)
b = np.where(m & n)[0]
indx = dataFinal.ix[a].index[b]
colorValues = dataFinal['obstemp'][indx]/32
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
c = plt.contourf(lons, lats, modLayerTemp, extend ='both')
clrmap  = c.cmap
plt.scatter(dataFinal['lon'][indx], dataFinal['lat'][indx], s=40, c=colorValues.values, cmap=clrmap)
ax[i].set_title('Layer: {0}'.format(l))
for i in range(1, 4):
    ax.append(plt.subplot(2,2,i+1))
    l = layer+i*4
    # lon, lat = dataFinal.ix[5]['lon'], dataFinal.ix[5]['lat']
    a = np.where(dataFinal['layer']==l)[0]
    m = dataFinal['time'][a]>starttime-timedelta(days=10)
    n = dataFinal['time'][a]<starttime+timedelta(days=10)
    b = np.where(m & n)[0]
    indx = dataFinal.ix[a].index[b]
    colorValues = dataFinal['obstemp'][indx]/32
    modLayerTemp = tempObj.layerTemp(l, url)  #grab new layer temp
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
    c = plt.contourf(lons, lats, modLayerTemp, extend ='both', cmap=clrmap)
    plt.scatter(dataFinal['lon'][indx], dataFinal['lat'][indx], s=40, c=colorValues.values, cmap=clrmap)
    ax[i].set_title('Layer: {0}'.format(l))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
plt.colorbar(c, cax=cbar_ax, ticks=range(0, 32, 4))     #c is the contour of first subplot
plt.suptitle('obsVSmodel, |obstemp-modemp|>10',fontsize=25)

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
plt.title('error bar that |obstemp-modtemp|>10, based on layers',fontsize=25)

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
plt.title('error bar that |obstemp-modtemp|>10, based on depth',fontsize=25)
plt.show()
