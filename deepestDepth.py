'''
return ratio of the deepest depth
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4
from datetime import datetime, timedelta
from turtleModule import str2ndlist
import watertempModule as wtm         # A module of classes that using ROMS, FVCOM.
obsData = pd.read_csv('ctd_good.csv') # From nearestIndexInMod.py
tf_index = np.where(obsData['TF'].notnull())[0] # Get  index of good data.
obsLat, obsLon = obsData['LAT'][tf_index], obsData['LON'][tf_index]
obsDeepest = obsData['MAX_DBAR'][tf_index] # get deepest data file depth
#obsID = obsData['PTT'][tf_index]           # Get ID of turtle.
layers = obsData['modDepthLayer'][tf_index]
modNearestIndex = pd.Series(str2ndlist(obsData['modNearestIndex'][tf_index], bracket=True), index=tf_index)

#starttime = datetime(2009, 8, 24)
starttime = datetime(2013,05,20)
endtime = datetime(2013, 12, 13)
tempObj = wtm.waterCTD()
url = tempObj.get_url(starttime, endtime)
modData = netCDF4.Dataset(url)
modTempAll = modData.variables['temp']
h = modData.variables['h']
newH=[]
for i in tf_index:
    m, n = int(modNearestIndex[i][0]), int(modNearestIndex[i][1])
    newH.append(h[m][n])
    print i

fig = plt.figure()
ax = fig.add_subplot(111)
p = obsDeepest/newH
# index1 = p[p>1.5].index
# id = obsID[index1]

y = np.arange(0,5,0.1)
x = np.array([0]*50)
for i in p:
    x[int(i*10)]+=1             # multiply 10 is because area 0.0~0.1 is 1st, 0.8~0.9 is 9th.
plt.barh(y, x,height=0.08)
plt.ylim(2.2,0)
plt.yticks(np.arange(0,5,0.1))
plt.ylabel('obsErrorDep/modH', fontsize=25)
plt.xlabel('Quantity', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
plt.title('Ratio of obs deepest', fontsize=25)
plt.show()
