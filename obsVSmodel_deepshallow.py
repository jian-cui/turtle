'''
draw the correlation of observation and model between deep and shallow(50m)'
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import netCDF4
from turtleModule import str2ndlist, np_datetime, bottom_value, closest_num
import watertempModule as wtm   # A module of classes that using ROMS, FVCOM.

def histogramPoints(x, y, bins):
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0, H)
    return xedges, yedges, Hmasked
def getModTemp(modTempAll, ctdTime, ctdLayer, ctdNearestIndex, s_rho, waterDepth, starttime, oceantime):
    '''
    Return model temp based on observation layers(commented out) or depth
    '''
    indx = closest_num((starttime -datetime(2006,01,01)).total_seconds(), oceantime)
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
        print 'ctdTime[i]', ctdTime[i]
        timeIndex1 = closest_num((ctdTime[i]-datetime(2006,01,01)).total_seconds(), oceantime)
        timeIndex = timeIndex1 - indx
        temp = modTempAll[timeIndex]
        temp[temp.mask] = 10000
        a, b = int(ctdNearestIndex[i][0]), int(ctdNearestIndex[i][1]) # index of nearest model node
        t = []
        for depth in ctdDepth[i]:
            depth = -depth
            locDepth = waterDepth[a, b]# Get the bottom depth of this location.
            lyrDepth = s_rho * locDepth# Depth of each layer
            print lyrDepth[-1], depth
            if depth > lyrDepth[-1]: # Obs is shallower than last layer which is the surface.
                d = (temp[-2,a,b]-temp[-1,a,b])/(lyrDepth[-2]-lyrDepth[-1]) * \
                    (depth-lyrDepth[-1]) + temp[-1,a,b]
            elif depth < lyrDepth[0]: # Obs is deeper than first layer which is the bottom.
                d = (temp[1,a,b]-temp[0,a,b])/(lyrDepth[1]-lyrDepth[0]) * \
                    (depth-lyrDepth[0]) + temp[0,a,b]
            else:
                ind = closest_num(depth, lyrDepth)
                d = (temp[ind,a,b]-temp[ind-1,a,b])/(lyrDepth[ind]-lyrDepth[ind-1]) * \
                    (depth-lyrDepth[ind-1]) + temp[ind-1,a,b]
            t.append(d)
        modTemp.append(t)
    modTemp = np.array(modTemp)
    return modTemp
#########################################MAIN CODE#####################################################################################################
FONTSIZE = 25
obs = pd.read_csv('ctdWithModTempByDepth.csv') # Extracted from "errorMap.py"
tf_index = np.where(obs['TF'].notnull())[0]    # Get the index of good data.
obsLon, obsLat = obs['LON'][tf_index], obs['LAT'][tf_index]
obsTime = pd.Series(np_datetime(obs['END_DATE'][tf_index]), index=tf_index)
obsTemp = pd.Series(str2ndlist(obs['TEMP_VALS'][tf_index]), index=tf_index)
# obsTemp = pd.Series(bottom_value(obs['TEMP_VALS'][tf_index]), index=tf_index)
obsDepth = pd.Series(str2ndlist(obs['TEMP_DBAR'][tf_index]), index=tf_index)
# obsDepth = obs['MAX_DBAR'][tf_index]
modLayer = pd.Series(str2ndlist(obs['modDepthLayer'][tf_index],bracket=True), index=tf_index) # bracket is to get rid of symbol "[" and "]" in string
modNearestIndex = pd.Series(str2ndlist(obs['modNearestIndex'][tf_index], bracket=True), index=tf_index)

starttime = datetime(2009, 8, 24)
endtime = datetime(2013, 12, 13)
# for depth
tempMod = pd.Series(str2ndlist(obs['modTempByDepth'][tf_index],bracket=True), index=tf_index)
'''
# for layers
tempObj = wtm.waterCTD()
url = tempObj.get_url(starttime, endtime)
# tempMod = tempObj.watertemp(obsLon.values, obsLat.values, obsDepth.values, obsTime.values, url)
modDataAll = tempObj.get_data(url)
oceantime = modDataAll['ocean_time']
modTempAll = modDataAll['temp']
tempMod = getModTemp(modTempAll, obsTime, modLayer, modNearestIndex, starttime, oceantime)

tempObsDeep, tempObsShallow = [], []
tempModDeep, tempModShallow = [], []
for i in range(len(obsTime.values)):
    for j in range(len(obsDepth.values[i])):
        print i, j
        if tempMod.values[i][j] > 100: continue
        if obsDepth.values[i][j] > 50.0:
            tempObsDeep.append(obsTemp.values[i][j])
            tempModDeep.append(tempMod.values[i][j])
        else:
            tempObsShallow.append(obsTemp.values[i][j])
            tempModShallow.append(tempMod.values[i][j])
'''
tempObs1, tempObs2, tempObs3, tempObs4 = [],[],[],[]
tempMod1, tempMod2, tempMod3, tempMod4 = [],[],[],[]
for i in range(len(obsTime.values)):
    for j in range(len(obsDepth.values[i])):
        print i, j
        d = obsDepth.values[i][j]
        if tempMod.values[i][j] > 100: continue
        if d<25.0:
            tempObs1.append(obsTemp.values[i][j])
            tempMod1.append(tempMod.values[i][j])
        elif d>=25.0 and d<50.0:
            tempObs2.append(obsTemp.values[i][j])
            tempMod2.append(tempMod.values[i][j])
        elif d>=50.0 and d<75:
            tempObs3.append(obsTemp.values[i][j])
            tempMod3.append(tempMod.values[i][j])
        elif d>=75.0:
            tempObs4.append(obsTemp.values[i][j])
            tempMod4.append(tempMod.values[i][j])
rng = ['25.0', '25.0~50.0', '50.0~75.0', '75.0']
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
x = obsLon
y = obsLat
z = obsDepth
# ax.scatter(x, y, z)
ax.contour(x, y, z)
ax.set_zlim([250,0])
plt.show()
'''
x = np.arange(0, 30, 0.01)
for i in range(4):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    exec('tempObs = tempObs%s' % str(i+1))
    exec('tempMod = tempMod%s' % str(i+1))
    x, y ,Hmasked = histogramPoints(tempObs, tempMod, 200)
    c = ax.pcolormesh(x, y, Hmasked)
    ax.plot(x, x, 'r-')
    fit = np.polyfit(tempObs, tempMod, 1)
    fit_fn = np.poly1d(fit)
    ax.plot(tempObs, fit_fn(tempObs), 'y-', linewidth=2)
    gradient, intercept, r_value, p_value, std_err\
         = stats.linregress(tempObs, tempMod)
    ax.set_title('%s, R-squard: %.4f' % (rng[i], r_value**2), fontsize=FONTSIZE)
    ax.set_xlabel('Obs temp', fontsize=FONTSIZE)
    ax.set_ylabel('Model temp', fontsize=FONTSIZE)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    cbar = plt.colorbar(c)
    cbar.ax.set_ylabel('Counts', fontsize=FONTSIZE)
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
'''
