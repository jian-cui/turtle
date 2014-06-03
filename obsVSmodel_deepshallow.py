import watertempModule as wtm
from  watertempModule import np_datetime, bottom_value
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from datetime import datetime, timedelta
from scipy import stats
def str2float(arg):
    ret = []
    for i in arg:
        i = i.split(',')
        b = []
        for j in i:
            j = float(j)
            b.append(j)
        ret.append(b)
    return ret
FONTSIZE = 25
ctd = pd.read_csv('ctd_extract_good.csv')
tf_index = np.where(ctd['TF'].notnull())[0]
ctdLon, ctdLat = ctd['LON'][tf_index], ctd['LAT'][tf_index]
ctdTime = pd.Series(np_datetime(ctd['END_DATE'][tf_index]), index=tf_index)
ctdTemp = pd.Series(str2float(ctd['TEMP_VALS'][tf_index]), index=tf_index)
# ctdTemp = pd.Series(bottom_value(ctd['TEMP_VALS'][tf_index]), index=tf_index)
ctdDepth = pd.Series(str2float(ctd['TEMP_DBAR'][tf_index]), index=tf_index)
# ctdDepth = ctd['MAX_DBAR'][tf_index]

starttime = datetime(2009, 8, 24)
endtime = datetime(2013, 12, 13)
tempObj = wtm.waterCTD()
url = tempObj.get_url(starttime, endtime)
tempMod = tempObj.watertemp(ctdLon.values, ctdLat.values, ctdDepth.values, ctdTime.values, url)

# dic = {'tempMod': tempMod, 'tempObs': ctdTemp, depth: ctdDepth}
# tempObs = pd.DataFrame(dic, index=tf_index)

tempObsShallow, tempObsShallow = [], []
tempModDeep, tempModDeep = [], []
for i in range(len(ctdTime.values))
    for j in range(len(ctdDepth.values[i])):
        print i, j
        if ctdDepth.values[i][j] > 50.0:
            tempObsDeep.append(ctdTemp[i][j])
            tempModDeep.append(tempMod[i][j])
        else:
            tempObsShallow.append(ctdTemp[i][j])
            tempModShallow.append(tempMod[i][j])

x = np.arange(0, 30, 0.01)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(tempObsDeep, tempModDeep, s=50, c='b')
ax1.plot(x, x, 'r-')
fit1 = np.ployfit(tempObsDeep, tempModDeep, 1)
fit_fn1 = np.ploy1d(fit1)
ax1.plot(tempObsDeep, fit_fn1(tempObsDeep), 'y--')
gradient1, intercept1, r_value1, p_value1, std_err1\
    = stats.linregress(tempObsDeep, tempModDeep)
ax1,set_title('>50m, R-squard: %.4f' % r_value1**2, fontsize=FONTSIZE)
ax1,set_xlabel('CTD temp', fontsize=FONTSIZE)
ax1.set_ylabel('Model temp', fontsize=FONTSIZE)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(tempObsShallow, tempModShallow)
ax2.plot(x, x, 'r-')
fit1 = np.ployfit(tempObsShallow, tempModShallow, 1)
fit_fn1 = np.ploy1d(fit1)
ax2.plot(tempObsShallow, fit_fn1(tempObsShallow), 'y--')
gradient1, intercept1, r_value1, p_value1, std_err1\
    = stats.linregress(tempObsShallow, tempModShallow)
ax2,set_title('>50m, R-squard: %.4f' % r_value1**2, fontsize=FONTSIZE)
ax2,set_xlabel('CTD temp', fontsize=FONTSIZE)
ax2.set_ylabel('Model temp', fontsize=FONTSIZE)
plt.show()
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
