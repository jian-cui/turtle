import watertempModule as wtm
from  watertempModule import np_datetime, bottom_value
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from datetime import datetime, timedelta
from scipy import stats
from module import str2ndlist, histogramPoints

def array_2dto1d(arr):
    arg = []
    for i in arr:
        for j in i:
            arg.append(j)
    arg = np.array(arg)
    return arg

FONTSIZE = 25
ctd = pd.read_csv('ctd_extract_good.csv', index_col=0)
tf_index = np.where(ctd['TF'].notnull())[0]
ctdLon, ctdLat = ctd['LON'][tf_index], ctd['LAT'][tf_index]
ctdTime = pd.Series(np_datetime(ctd['END_DATE'][tf_index]), index=tf_index)
ctdTemp = pd.Series(str2ndlist(ctd['TEMP_VALS'][tf_index]), index=tf_index)
# ctdTemp = pd.Series(bottom_value(ctd['TEMP_VALS'][tf_index]), index=tf_index)
ctdDepth = pd.Series(str2ndlist(ctd['TEMP_DBAR'][tf_index]), index=tf_index)
ctdMaxDepth = ctd['MAX_DBAR'][tf_index]

starttime = datetime(2009, 8, 24)
endtime = datetime(2013, 12, 13)
tempObj = wtm.waterCTD()
url = tempObj.get_url(starttime, endtime)
tempMod = np.array(tempObj.watertemp(ctdLon.values, ctdLat.values, ctdDepth.values, ctdTime.values, url))

# dic = {'tempMod': tempMod, 'tempObs': ctdTemp, depth: ctdDepth}
# tempObs = pd.DataFrame(dic, index=tf_index)

tempObsDeep, tempObsShallow = [], []
tempModDeep, tempModShallow = [], []
i = ctdMaxDepth.values>50
tempObsDeep = array_2dto1d(ctdTemp.values[i])
tempModDeep = array_2dto1d(tempMod[i])
tempObsShallow = array_2dto1d(ctdTemp.values[~i])
tempModShallow = array_2dto1d(tempMod[~i])

nbins = 200
x = np.arange(0, 30, 0.01)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
# ax1.scatter(tempObsDeep, tempModDeep, s=50, c='b')
ax1.plot(x, x, 'r-')
fit1 = np.polyfit(tempObsDeep, tempModDeep, 1)
fit_fn1 = np.poly1d(fit1)
ax1.plot(tempObsDeep, fit_fn1(tempObsDeep), 'y-')
gradient1, intercept1, r_value1, p_value1, std_err1\
    = stats.linregress(tempObsDeep, tempModDeep)
xe, ye, Hmasked = histogramPoints(tempObsDeep, tempModDeep, nbins)
plt.pcolormesh(xe, ye, Hmasked)
ax1.set_title('offshore, R-squard: %.4f' % r_value1**2, fontsize=FONTSIZE)
ax1.set_xlabel('CTD temp', fontsize=FONTSIZE)
ax1.set_ylabel('Model temp', fontsize=FONTSIZE)
ax1.axis([0, 35, 0,35])
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts', fontsize=FONTSIZE)

i = np.where(np.array(tempModShallow)<100) #Some of the data is infinity.
tempObsShallow1 = np.array(tempObsShallow)[i]
tempModShallow1 = np.array(tempModShallow)[i]
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
# ax2.scatter(tempObsShallow1, tempModShallow1)
xe, ye, Hmasked = histogramPoints(tempObsShallow1, tempModShallow1, nbins)
plt.pcolormesh(xe, ye, Hmasked)
ax2.plot(x, x, 'r-')
fit2 = np.polyfit(tempObsShallow1, tempModShallow1, 1)
fit_fn2 = np.poly1d(fit2)
ax2.plot(tempObsShallow1, fit_fn2(tempObsShallow1), 'y-')
gradient2, intercept2, r_value2, p_value2, std_err2\
    = stats.linregress(tempObsShallow1, tempModShallow1)
ax2.set_title('onshore, R-squard: %.4f' % r_value2**2, fontsize=FONTSIZE)
ax2.set_xlabel('CTD temp', fontsize=FONTSIZE)
ax2.set_ylabel('Model temp', fontsize=FONTSIZE)
ax2.axis([0, 35, 0,35])
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts', fontsize=FONTSIZE)
plt.show()


#plot all points
tempObsn = array_2dto1d(ctdTemp)
tempModn = array_2dto1d(tempMod)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
# ax2.scatter(tempObsShallow1, tempModShallow1)
xe, ye, Hmasked = histogramPoints(tempObsn, tempModn, nbins)
plt.pcolormesh(xe, ye, Hmasked)
ax3.plot(x, x, 'r-')
fit3 = np.polyfit(tempObsn, tempModn, 1)
fit_fn3 = np.poly1d(fit3)
ax3.plot(tempObsn, fit_fn3(tempObsn), 'y-')
gradient3, intercept3, r_value3, p_value3, std_err3\
    = stats.linregress(tempObsn, tempModn)
ax3.set_title('all, R-squard: %.4f' % r_value3**2, fontsize=FONTSIZE)
ax3.set_xlabel('CTD temp', fontsize=FONTSIZE)
ax3.set_ylabel('Model temp', fontsize=FONTSIZE)
ax3.axis([0, 35, 0,35])
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts', fontsize=FONTSIZE)
plt.show()
