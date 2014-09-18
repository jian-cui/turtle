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
        b = np.array([float(j) for j in a])
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
def getModTemp(modTempAll, ctdTime, ctdLayer, ctdNearestIndex, s_rho, waterDepth, starttime, oceantime):
    ind = closest_num((starttime -datetime(2006,01,01)).total_seconds(), oceantime)
    modTemp = []
    l = len(ctdLayer.index)
    for i in ctdLayer.index:
        '''
        # For layers
        print i, l, 'getModTemp'
        timeIndex = closest_num((ctdTime[i]-datetime(2006,01,01)).total_seconds(), oceantime)-ind
        modTempTime = modTempAll[timeIndex]
        modTempTime[modTempTime.mask] = 10000
        t = np.array([modTempTime[ctdLayer[i][j],ctdNearestIndex[i][0], ctdNearestIndex[i][1]] \
                          for j in range(len(ctdLayer[i]))])
        modTemp.append(t)
        '''
        # For depth
        print i, l, 'getModTemp'
        timeIndex = closest_num((ctdTime[i]-datetime(2006,01,01)).total_seconds(), oceantime)-ind
        temp = modTempAll[timeIndex]
        a, b = int(ctdNearestIndex[i][0]), int(ctdNearestIndex[i][1])
        t = []
        for depth in ctdDepth[i]:
            locDepth = waterDepth[a, b]
            lyrDepth = s_rho * locDepth
            if depth > lyrDepth[-1]: # Obs is shallower than last layer.
                d = (temp[-2,a,b]-temp[-1,a,b])/(lyrDepth[-2]-lyrDepth[-1]) * \
                    (depth-lyrDepth[-1]) + temp[-1,a,b]
            elif depth < lyrDepth[0]: # Obs is deeper than first layer.
                d = (temp[1,a,b]-temp[0,a,b])/(lyrDepth[1]-lyrDepth[0]) * \
                    (depth-lyrDepth[0]) + temp[0,a,b]
            else:
                ind = self.closest_num(depth, lyrDepth)
                d = (temp[ind,a,b]-temp[ind-1,a,b])/(lyrDepth[ind,a,b]-lyrDepth[ind-1,a,b]) * \
                    (depth-lyrDepth[ind-1]) + temp[ind-1,a,b]
            t.append(d)
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
s_rho = modDataAll['s_rho']
waterDepth = modDataAll['h']
modTemp = getModTemp(modTempAll, ctdTime, ctdLayer, ctdNearestIndex, s_rho, waterDepth, starttime, oceantime)

data = pd.DataFrame({'lon': ctdLon, 'lat': ctdLat,
                     'obstemp': ctdTemp.values,'modtemp':modTemp,
                     'depth': ctdDepth, 'time': ctdTime.values,
                     'layer': ctdLayer}, index=tf_index)

ind = [] # the indices needed
obst = []
modt = []
lyr = []
dep=[]
text = '|modtemp-obstemp|>10 degC' # needed to be consistent with the "if" statement below!!!!!!
for i in data.index:
    print i
    diff = abs(data['obstemp'][i] - data['modtemp'][i])
    indx = np.where(diff > 10)[0]
    if not indx.size: continue
    ind.extend([i] * indx.size)
    obst.extend(data['obstemp'][i][indx])
    modt.extend(np.array(data['modtemp'][i])[indx])
    lyr.extend(np.array(data['layer'][i])[indx])
    dep.extend(np.array(data['depth'][i])[indx])
    '''
    for j in range(len(a['obstemp'][i])):
        print i, j
        y = data['modtemp'][i][j]
        x = data['obstemp'][i][j]
        if abs(x - y) > 10:     # |mod-obstemp|>10
        # if y - x > 10:          # modtemp-obstemp>10
        # if x - y > 10:          # obstemp-modtenp>10
            ind.append(i)
            obst.append(x)
            modt.append(y)
            lyr.append(data['layer'][i][j])
            dep.append(data['depth'][i][j])
    '''
dataFinal = pd.DataFrame({'lon': data['lon'][ind].values,
                          'lat': data['lat'][ind].values,
                          'time': data['time'][ind].values,
                          'obstemp': np.array(obst),
                          'modtemp': np.array(modt),
                          'layer': np.array(lyr),
                          'dep': np.array(dep),
                          'nearestIndex': ctdNearestIndex[ind].values
                          })
starttime = datetime(2013,07,10)
endtime = starttime + timedelta(hours=1)
# layer = 15
depth = -10

tempObj = wtm.water_roms()
url = tempObj.get_url(starttime, endtime)
modData = tempObj.get_data(url)
h, s_rho = modData['h'], modData['s_rho']
lons, lats = modData['lon_rho'], modData['lat_rho']
'''
#get the mod layer. Unuseful after using "ctd_good.csv".
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
i = 0
for i in range(0, 4):
    ax.append(plt.subplot(2,2,i+1))
    # l = layer+i*6
    l = depth - i*25
    # lon, lat = dataFinal.ix[5]['lon'], dataFinal.ix[5]['lat']
    # p = np.where(dataFinal['layer']==l)[0]
    a = dataFinal['dep'] < 5-l
    b = dataFinal['dep'] > -5-l
    p = np.where(a & b)[0]
    m = dataFinal['time'][p]>starttime-timedelta(days=10)
    n = dataFinal['time'][p]<starttime+timedelta(days=10)
    b = np.where(m & n)[0]
    indx = dataFinal.ix[p].index[b]
    colorValues = dataFinal['obstemp'][indx]/32
    # modLayerTemp = tempObj.layerTemp(l, url)  #grab new layer temp
    modLayerTemp = tempObj.depthTemp(l, url)
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
    if i == 0:
        c = plt.contourf(lons, lats, modLayerTemp, extend ='both')
        clrmap = c.cmap
    else:
        c = plt.contourf(lons, lats, modLayerTemp, extend ='both', cmap=clrmap)
    plt.scatter(dataFinal['lon'][indx], dataFinal['lat'][indx], s=40, c=colorValues.values, cmap=clrmap)
    ax[i].set_title('Depth: {0}'.format(l))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
plt.colorbar(c, cax=cbar_ax, ticks=range(0, 32, 4))     #c is the contour of first subplot
plt.suptitle('obsVSmodel, %s' % text, fontsize=25)
'''
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
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('error bar that %s, based on layers' % text, fontsize=25)
'''
#draw errorbar based on depth.
fig = plt.figure()
ax = fig.add_subplot(111)
y = dataFinal['dep'].order().values
x = np.arange(1, np.amax(y)+1)
bar = np.array([0]*np.amax(y))
for i in y:
    # if i in x:
    bar[int(i)-1] = bar[int(i)-1]+1
plt.barh(x, bar)
plt.ylim(50, 0)
# plt.ylabel('depth', fontsize=25)
plt.ylabel('dpeth(meters)', fontsize=25)
plt.xlabel('Quantity', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title('error bar that |obstemp-modtemp|>10, based on depth',fontsize=25)
plt.title('%s' % text, fontsize=25)

modDepth = []
for i in dataFinal.index:
    m = dataFinal['nearestIndex'][i]
    modDepth.append(h[int(m[0]), int(m[1])])
fig=plt.figure()
ax = fig.add_subplot(111)
rate = dataFinal['dep']/modDepth
x = [0]*50
y = np.arange(0,5,0.1)
for i in rate:
    x[int(i*10)] += 1
plt.barh(y, x, height=0.08)
plt.ylim(5, 0)
plt.yticks(np.arange(0,5,0.1))
plt.ylabel('obsErrorDep/modH', fontsize=25)
plt.xlabel('Quantity', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Ratio of obs error(>10)', fontsize=25)
plt.show()

