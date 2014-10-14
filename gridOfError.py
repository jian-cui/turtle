import netCDF4
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import pandas as pd
from module import str2ndlist, str2float
from watertempModule import np_datetime
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
                       labels=[1,0,0,0], linewidth=0)
    dmap.drawmeridians(np.arange(int(min(lonsize))-1,
                                 int(max(lonsize))+1,interval_lon),
                       labels=[0,0,0,1], linewidth=0)
    dmap.drawcoastlines()
    dmap.fillcontinents(color='grey')
    dmap.drawmapboundary()
def intersection(l1, l2):
    '''
    Calculate point of intersection of two lines.
    line1: y = k1*x + b1
    line2: y = k2*x + b2
    x, y = intersection((k1, b1), (k2, b2))
    '''
    k1, b1 = l1[0], l1[1]
    k2, b2 = l2[0], l2[1]
    x = (b2-b1)/(k1-k2)
    y = k1*x + b1
    return x, y
def whichArea(arg, lst):
    i = len(lst)//2
    if i != 0: 
        if arg >= lst[i]:#!!!!!!!!!!!!!!!!wrong!!!!!!!!!!!!
            r = i + whichArea(arg, lst[i:])
        elif arg < lst[i]:
            r = whichArea(arg, lst[:i])
    else: r = i
    return r
    
url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/hidden/2006_da/his?lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],u[0:1:69911][0:1:35][0:1:81][0:1:128],v[0:1:69911][0:1:35][0:1:80][0:1:129]'
data = netCDF4.Dataset(url)
lons, lats = data.variables['lon_rho'], data.variables['lat_rho']

lonA, latA = lons[81][0], lats[81][0]
lonB, latB = lons[81][129], lats[81][129]
lonC, latC = lons[0][129], lats[0][129]
lonD, latD = lons[0][0], lats[0][0]

ctdData = pd.read_csv('ctdWithModTempByDepth.csv')
tf_index = np.where(ctdData['TF'].notnull())[0]
ctdNearestIndex = pd.Series(str2ndlist(ctdData['modNearestIndex'][tf_index], bracket=True), index=tf_index)
modTemp = pd.Series(str2ndlist(ctdData['modTempByDepth'][tf_index],bracket=True), index=tf_index)
ctdLon, ctdLat = ctdData['LON'][tf_index], ctdData['LAT'][tf_index]
ctdTime = pd.Series(np_datetime(ctdData['END_DATE'][tf_index]), index=tf_index)
ctdTemp = pd.Series(str2float(ctdData['TEMP_VALS'][tf_index]), index=tf_index)


data = pd.DataFrame({'lon': ctdLon, 'lat': ctdLat,
                     'obstemp': ctdTemp.values,'modtemp':modTemp,
                     'time': ctdTime.values, 'nearestIndex': ctdNearestIndex.values},
                    index=tf_index)

lonsize = [np.amin(lons), np.amax(lons)]
latsize = [np.amin(lats), np.amax(lats)]

errorNum = []
for i in range(9):
    j = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    errorNum.append(j)
fig = plt.figure()
ax = fig.add_subplot(111)
draw_basemap(fig, ax, lonsize, latsize)
plt.plot([lonA, lonB], [latA, latB], 'b-')
plt.plot([lonD, lonC], [latD, latC], 'b-')
plt.plot([lonD, lonA], [latD, latA], 'b-')
plt.plot([lonC, lonB], [latC, latB], 'b-')
for i in range(0, 75, 10):      # Here use num smller than 81 because the last grid is too small
    plt.plot([lons[i][0], lons[i][129]], [lats[i][0], lats[i][129]], 'b--')
for i in range(0, 129, 10):
    plt.plot([lons[0][i], lons[81][i]], [lats[0][i], lats[81][i]], 'b--')
r1 = range(0, 81, 10)
r2 = range(0, 129, 10)
nearestIndex = []
for i in data.index:
    diff = data['obstemp'][i] - data['modtemp'][i]
    indx = np.where(abs(diff)>10)[0]
    if not indx.size: continue
    nearestIndex.extend([ctdNearestIndex[i]] * indx.size)
    '''
    # all points
    m = len(data['lon'])
    nearestIndex.extend([ctdNearestIndex[i]] * m)
    '''
for i in nearestIndex:
    m = whichArea(i[0], r1)
    n = whichArea(i[1], r2)
    errorNum[m][n] += 1
m1, m2 = 34.05, 39.84
n1, n2 = -75.83, -67.72
for s in range(8):
# a = np.arange(-75.83, -67.72, 0.631)
# b = np.arange(34.05, 39.84, 0.47)
    a = np.arange(n1, n2, 0.631)
    b = np.arange(m1, m2, 0.47)
    for i, j, k in zip(a, b, errorNum[s]):
        print i, j, k
        plt.text(i, j, str(k), color='r',multialignment='center')
    m1 = m1 + 0.408
    m2 = m2 + 0.408
    n1 = n1 - 0.45
    n2 = n2 - 0.45
plt.title('Distribution of Error', fontsize=30)

#########################################################
dataNum = []
for i in range(9):
    j = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    dataNum.append(j)
fig = plt.figure()
ax = fig.add_subplot(111)
draw_basemap(fig, ax, lonsize, latsize)
plt.plot([lonA, lonB], [latA, latB], 'b-')
plt.plot([lonD, lonC], [latD, latC], 'b-')
plt.plot([lonD, lonA], [latD, latA], 'b-')
plt.plot([lonC, lonB], [latC, latB], 'b-')
for i in range(0, 75, 10):      # Here use num smller than 81 because the last grid is too small
    plt.plot([lons[i][0], lons[i][129]], [lats[i][0], lats[i][129]], 'b--')
for i in range(0, 129, 10):
    plt.plot([lons[0][i], lons[81][i]], [lats[0][i], lats[81][i]], 'b--')
r1 = range(0, 81, 10)
r2 = range(0, 129, 10)
nearestIndex = []
for i in data.index:
    # all points
    m = len(data['lon'])
    nearestIndex.extend([ctdNearestIndex[i]] * m)
for i in nearestIndex:
    m = whichArea(i[0], r1)
    n = whichArea(i[1], r2)
    dataNum[m][n] += 1
m1, m2 = 34.05, 39.84
n1, n2 = -75.83, -67.72
for s in range(8):
    a = np.arange(n1, n2, 0.631)
    b = np.arange(m1, m2, 0.47)
    for i, j, k in zip(a, b, dataNum[s]):
        print i, j, k
        plt.text(i, j, str(k), color='r',multialignment='center', ha='center')
    m1 = m1 + 0.408
    m2 = m2 + 0.408
    n1 = n1 - 0.45
    n2 = n2 - 0.45
plt.title('Distribution of Data', fontsize=30)
plt.show()
