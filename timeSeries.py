'''
draw temp change of one specific turtle data and model data.
'''
import numpy as np
import pandas as pd
from module import str2ndlist
from  watertempModule import np_datetime, bottom_value, dist
import watertempModule as wtm
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta
import netCDF4
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
    ind = closest_num((starttime -datetime(2013,05,18)).total_seconds()/3600, oceantime)
    modTemp = []
    l = len(ctdLayer.index)
    for i in ctdLayer.index:
        print i, l
        timeIndex = closest_num((ctdTime[i]-datetime(2013,05,18)).total_seconds()/3600, oceantime)-ind
        modTempTime = modTempAll[timeIndex]
        # modTempTime[modTempTime.mask] = 10000
        t = [modTempTime[ctdLayer[i][j],ctdNearestIndex[i][0], ctdNearestIndex[i][1]] \
             for j in range(len(ctdLayer[i]))]
        modTemp.append(t)
    modTemp = np.array(modTemp)
    return modTemp
def smooth(v, e):
    #v should be a list
    for i in range(len(v))[1:-1]:
        a = v[i]
        b = v[i+1]
        c = v[i-1]
        diff1 = abs(a - b)
        diff2 = abs(a - c)
        if diff2>e:
            v[i] = c
    return v
ctdData = pd.read_csv('ctd_good.csv', index_col=0)
tf_index = np.where(ctdData['TF'].notnull())[0]
ctdData = ctdData.ix[tf_index]
id = ctdData['PTT'].drop_duplicates().values
tID = id[6]  #0~4, 6,7,8,9
layers = pd.Series(str2ndlist(ctdData['modDepthLayer'], bracket=True), index=ctdData.index)
locIndex = pd.Series(str2ndlist(ctdData['modNearestIndex'], bracket=True), index=ctdData.index)
ctdTemp = pd.Series(str2ndlist(ctdData['TEMP_VALS'].values), index=ctdData.index)
ctdTime = pd.Series(np_datetime(ctdData['END_DATE'].values), index=ctdData.index)

layers = layers[ctdData['PTT']==tID]
locIndex = locIndex[ctdData['PTT']==tID]
time = ctdTime[ctdData['PTT']==tID]
temp = ctdTemp[ctdData['PTT']==tID]

starttime, endtime=np.amin(time), np.amax(time)+timedelta(hours=1)
modObj = wtm.waterCTD()
url = modObj.get_url(starttime, endtime)
oceantime = netCDF4.Dataset(url).variables['ocean_time']
modTempAll = netCDF4.Dataset(url).variables['temp']
modTemp = getModTemp(modTempAll, ctdTime, layers, locIndex, starttime, oceantime)
modTemp = pd.Series(modTemp, index=temp.index)

obsMaxTemp, obsMinTemp = [], []
modMaxTemp, modMinTemp = [], []
for i in temp.index:
    obsMaxTemp.append(max(temp[i]))
    obsMinTemp.append(min(temp[i]))
    modMaxTemp.append(max(modTemp[i]))
    modMinTemp.append(min(modTemp[i]))
data = pd.DataFrame({'time':time.values, 'obsMaxTemp':obsMaxTemp, 'obsMinTemp':obsMinTemp,
                    'modMaxTemp': modMaxTemp, 'modMinTemp': modMinTemp}, index=range(len(time)))
data = data.sort_index(by='time')

data['obsMinTemp'] = smooth(data['obsMinTemp'].values, 5)
data['modMinTemp'] = smooth(data['modMinTemp'].values, 5)
data['time'] = smooth(data['time'].values, timedelta(days=20))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data['time'], data['obsMaxTemp'], color='b', linewidth=2)
ax.plot(data['time'], data['obsMinTemp'], color='b', linewidth=2, label='obs')
ax.plot(data['time'], data['modMaxTemp'], color='r', linewidth=2)
ax.plot(data['time'], data['modMinTemp'], color='r', linewidth=2, label='mod')
plt.legend()
ax.set_xlabel('time', fontsize=20)
ax.set_ylabel('temperature', fontsize=20)
dates = mpl.dates.drange(np.amin(time), np.max(time), timedelta(days=30))
dateFmt = mpl.dates.DateFormatter('%b,%Y')
ax.set_xticks(dates)
ax.xaxis.set_major_formatter(dateFmt)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('time series of temp for turtle:{0}'.format(tID), fontsize=25)
plt.show()
'''
fig = plt.figure()
ax2 = fig.add_subplot(111)
for i in temp.index:
    # ax2.plot(([time[i]+timedelta(hours=5)])*len(temp[i]), temp[i],color='b')
    ax2.plot([time[i]]*len(temp[i]), modTemp[i], color='r')
ax2.set_xlabel('time', fontsize=20)
ax2.set_ylabel('temperature', fontsize=20)
dates = mpl.dates.drange(np.amin(time), np.max(time), timedelta(days=30))
dateFmt = mpl.dates.DateFormatter('%b,%Y')
ax2.set_xticks(dates)
ax2.xaxis.set_major_formatter(dateFmt)
ax2.set_title('mod', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

ax1 = fig.add_subplot(211)
for i in temp.index:
    ax1.plot([time[i]]*len(temp[i]), temp[i], color='b')
ax1.set_ylabel('temperature', fontsize=20)
ax1.set_xticks(dates)
ax1.xaxis.set_major_formatter(dateFmt)
ax1.set_title('obs', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
fig.suptitle('time series of temp for turtle:{0}'.format(tID), fontsize=25)
ax2.set_yticks(ax1.get_yticks())
plt.show()
'''
