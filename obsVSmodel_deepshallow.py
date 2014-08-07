import watertempModule as wtm
from  watertempModule import np_datetime, bottom_value
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from datetime import datetime, timedelta
from scipy import stats
from module import str2ndlist
import netCDF4

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
def histogramPoints(x, y, bins):
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0, H)
    return xedges, yedges, Hmasked
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
        t = [modTempTime[ctdLayer[i][j],ctdNearestIndex[i][0], ctdNearestIndex[i][1]] \
             for j in range(len(ctdLayer[i]))]
        modTemp.append(t)
    modTemp = np.array(modTemp)
    return modTemp
FONTSIZE = 25
ctd = pd.read_csv('ctd_good.csv')
tf_index = np.where(ctd['TF'].notnull())[0]
ctdLon, ctdLat = ctd['LON'][tf_index], ctd['LAT'][tf_index]
ctdTime = pd.Series(np_datetime(ctd['END_DATE'][tf_index]), index=tf_index)
ctdTemp = pd.Series(str2float(ctd['TEMP_VALS'][tf_index]), index=tf_index)
# ctdTemp = pd.Series(bottom_value(ctd['TEMP_VALS'][tf_index]), index=tf_index)
ctdDepth = pd.Series(str2float(ctd['TEMP_DBAR'][tf_index]), index=tf_index)
# ctdDepth = ctd['MAX_DBAR'][tf_index]
ctdLayer = pd.Series(str2ndlist(ctd['modDepthLayer'][tf_index],bracket=True), index=tf_index)
ctdNearestIndex = pd.Series(str2ndlist(ctd['modNearestIndex'][tf_index], bracket=True), index=tf_index)

starttime = datetime(2009, 8, 24)
endtime = datetime(2013, 12, 13)
tempObj = wtm.waterCTD()
url = tempObj.get_url(starttime, endtime)
# tempMod = tempObj.watertemp(ctdLon.values, ctdLat.values, ctdDepth.values, ctdTime.values, url)
modDataAll = tempObj.get_data(url)
oceantime = modDataAll['ocean_time']
modTempAll = modDataAll['temp']
tempMod = getModTemp(modTempAll, ctdTime, ctdLayer, ctdNearestIndex, starttime, oceantime)

# dic = {'tempMod': tempMod, 'tempObs': ctdTemp, depth: ctdDepth}
# tempObs = pd.DataFrame(dic, index=tf_index)

tempObsDeep, tempObsShallow = [], []
tempModDeep, tempModShallow = [], []
for i in range(len(ctdTime.values)):
    for j in range(len(ctdDepth.values[i])):
        print i, j
        if ctdDepth.values[i][j] > 50.0:
            tempObsDeep.append(ctdTemp.values[i][j])
            tempModDeep.append(tempMod[i][j])
        else:
            tempObsShallow.append(ctdTemp.values[i][j])
            tempModShallow.append(tempMod[i][j])
'''
#use scatter
x = np.arange(0, 30, 0.01)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(tempObsDeep, tempModDeep, s=50, c='b')
ax1.plot(x, x, 'r-')
fit1 = np.polyfit(tempObsDeep, tempModDeep, 1)
fit_fn1 = np.poly1d(fit1)
ax1.plot(tempObsDeep, fit_fn1(tempObsDeep), 'y-')
gradient1, intercept1, r_value1, p_value1, std_err1\
    = stats.linregress(tempObsDeep, tempModDeep)
ax1.set_title('>50m, R-squard: %.4f' % r_value1**2, fontsize=FONTSIZE)
ax1.set_xlabel('CTD temp', fontsize=FONTSIZE)
ax1.set_ylabel('Model temp', fontsize=FONTSIZE)

i = np.where(np.array(tempModShallow)<100) #Some of the data is infinity.
tempObsShallow1 = np.array(tempObsShallow)[i]
tempModShallow1 = np.array(tempModShallow)[i]
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(tempObsShallow1, tempModShallow1)
ax2.plot(x, x, 'r-')
fit2 = np.polyfit(tempObsShallow1, tempModShallow1, 1)
fit_fn2 = np.poly1d(fit2)
ax2.plot(tempObsShallow1, fit_fn2(tempObsShallow1), 'y-')
gradient2, intercept2, r_value2, p_value2, std_err2\
    = stats.linregress(tempObsShallow1, tempModShallow1)
ax2.set_title('<50m, R-squard: %.4f' % r_value2**2, fontsize=FONTSIZE)
ax2.set_xlabel('CTD temp', fontsize=FONTSIZE)
ax2.set_ylabel('Model temp', fontsize=FONTSIZE)
plt.show()
'''
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
x = ctdLon
y = ctdLat
z = ctdDepth
# ax.scatter(x, y, z)
ax.contour(x, y, z)
ax.set_zlim([250,0])
plt.show()
'''
#use histogram2d and pcolormesh
x = np.arange(0, 30, 0.01)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
# ax1.scatter(tempObsDeep, tempModDeep, s=50, c='b')
x1, y1, Hmasked1 = histogramPoints(tempObsDeep, tempModDeep, 200)
c1 = ax1.pcolormesh(x1, y1, Hmasked1)
ax1.plot(x, x, 'r-')
fit1 = np.polyfit(tempObsDeep, tempModDeep, 1)
fit_fn1 = np.poly1d(fit1)
ax1.plot(tempObsDeep, fit_fn1(tempObsDeep), 'y-', linewidth=2)
gradient1, intercept1, r_value1, p_value1, std_err1\
    = stats.linregress(tempObsDeep, tempModDeep)
ax1.set_title('Deep(>50m), R-squard: %.4f' % r_value1**2, fontsize=FONTSIZE)
ax1.set_xlabel('Obs temp', fontsize=FONTSIZE)
ax1.set_ylabel('Model temp', fontsize=FONTSIZE)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar = plt.colorbar(c1)
cbar.ax.set_ylabel('Counts', fontsize=FONTSIZE)

i = np.where(np.array(tempModShallow)<100) #Some of the data is infinity.
tempObsShallow1 = np.array(tempObsShallow)[i]
tempModShallow1 = np.array(tempModShallow)[i]
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
# ax2.scatter(tempObsShallow1, tempModShallow1)
x2, y2, Hmasked2 = histogramPoints(tempObsShallow1, tempModShallow1, 200)
c2 = ax2.pcolormesh(x2, y2, Hmasked2)
ax2.plot(x, x, 'r-')
fit2 = np.polyfit(tempObsShallow1, tempModShallow1, 1)
fit_fn2 = np.poly1d(fit2)
ax2.plot(tempObsShallow1, fit_fn2(tempObsShallow1), 'y-', linewidth=2)
gradient2, intercept2, r_value2, p_value2, std_err2\
    = stats.linregress(tempObsShallow1, tempModShallow1)
ax2.set_title('Shallow(<50m), R-squard: %.4f' % r_value2**2, fontsize=FONTSIZE)
ax2.set_xlabel('Obs temp', fontsize=FONTSIZE)
ax2.set_ylabel('Model temp', fontsize=FONTSIZE)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar = plt.colorbar(c2)
cbar.ax.set_ylabel('Count', fontsize=FONTSIZE)
cbar.ax.set_yticks(fontsize=20)
plt.show()
