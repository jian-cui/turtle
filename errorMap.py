import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from time import strptime
import watertempModule as wtm
# d = {'lon': ctdLon.values, 'lat': ctdLat.values, 'obstemp': ctdTemp.values,
#      'modtemp':tempMod, 'depth': ctdDepth, 'time': ctdTime.values}
# a = pd.DataFrame(d, index=tf_index)
a = pd.read_csv('temp.csv',index_col=0)
modTime = []
for i in a['time']:
    modTime.append(strptime(i, '%Y-%m-%d %H:%M:%S'))
modTime = pd.Series(modTime, index=a.index)
ind = [] # the indices needed
obst = []
modt = []
dep = []
for i in a.index:
    # obst = []
    # modt = []
    # dep = []
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
                          'time': modTime[ind].values,
                          'obstemp': np.array(obst),
                          'modtemp': np.array(modt),
                          'dep': np.array(dep)
                          })
starttime = datetime(2013,07,07,04)
endtime = starttime + timedelta(hours=1)
depth = 16
tempObj = wtm.water_roms()
url = tempObj.get_url(starttime, endtime)
modTemp = tempObj.layerTemp(lon, lat, depth, url)

lonsize = np.amin(d['lon'])-0.1, np.amax(d['lon'])+0.1
latsize = np.amin(d['lat'])-0.1, np.amax(d['lat'])+0.1
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
plt.contourf(objData['lon'], objData['lat'], modTemp, extend='both')

