'''
plot 4 maps in 1 figure to show which depth has the most errors. Also plot the errorbar and ratio
'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import watertempModule as wtm
from  watertempModule import np_datetime, bottom_value, dist
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from module import str2ndlist, str2float
import netCDF4

FONTSIZE = 25
obsData = pd.read_csv('ctdWithModTempByDepth.csv') # extracted from ctdWithModTempByDepth.py
tf_index = np.where(obsData['TF'].notnull())[0]    # get the index of good data
obsLon, obsLat = obsData['LON'][tf_index], obsData['LAT'][tf_index]
obsTime = pd.Series(np_datetime(obsData['END_DATE'][tf_index]), index=tf_index)
obsTemp = pd.Series(str2float(obsData['TEMP_VALS'][tf_index]), index=tf_index)
# obsTemp = pd.Series(bottom_value(obs['TEMP_VALS'][tf_index]), index=tf_index)
obsDepth = pd.Series(str2float(obsData['TEMP_DBAR'][tf_index]), index=tf_index)
modLayer = pd.Series(str2ndlist(obsData['modDepthLayer'][tf_index],bracket=True), index=tf_index) # bracket is to get rid of symbol "[" and "]" in string
modNearestIndex = pd.Series(str2ndlist(obsData['modNearestIndex'][tf_index], bracket=True), index=tf_index)
modTemp = pd.Series(str2ndlist(obsData['modTempByDepth'][tf_index], bracket=True), index=tf_index)

data = pd.DataFrame({'lon': obsLon, 'lat': obsLat,
                     'obstemp': obsTemp.values,'modtemp':modTemp,
                     'depth': obsDepth, 'time': obsTime.values,
                     'layer': modLayer}, index=tf_index)

ind = [] # the indices needed
obst = []
modt = []
lyr = []
dep=[]
text = '|modtemp-obstemp|>10 degC' # remember to keep consistent with the "if" statement below
for i in data.index:
    diff = data['obstemp'][i] - data['modtemp'][i]
    indx = np.where(abs(diff) > 10)[0]
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
                          'nearestIndex': modNearestIndex[ind].values
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
        plt.contourf(lons, lats, modLayerTemp, extend ='both', cmap=clrmap)
    plt.scatter(dataFinal['lon'][indx], dataFinal['lat'][indx], s=40, c=colorValues.values, cmap=clrmap)
    ax[i].set_title('Depth: {0}'.format(l))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
plt.colorbar(c, cax=cbar_ax, ticks=range(0, 32, 4))     #c is the contour of first subplot
plt.suptitle('obsVSmodel, %s' % text, fontsize=25)
'''
# for layer
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
plt.title('%s(depth)' % text, fontsize=25)

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
plt.title('Ratio of obs error(>10) (depth)', fontsize=25)
plt.show()

