'''
Draw the errorMap of off-shore and on-shore
'''
import watertempModule as wtm
from watertempModule import np_datetime
import numpy as np
import matplotlib as mpl
import pandas as pd
from module import str2ndlist
from datetime import datetime, timedelta

def closest_num(self, num, numlist, i=0):
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
        i = self.closest_num(num, numlist[indx:],
                          i=i+indx)
    elif num < numlist[indx]:
        i = self.closest_num(num, numlist[0:indx+1], i=i)
    return i
def getModTemp(modTempAll, ctdTime, ctdLayer, ctdNeareastIndex):
    ind = closest_num(t1, self.oceantime)
    modTemp = []
    for i in ctdLayer.index:
        timeIndex = cloest_num(ctdTime[i]-datetime(2006,01,01).total_seconds, oceantime)-ind
        modTempAll[modTempAll.mask] = 10000
        t = [modTempAll[timeIndex,ctdLayer[i][j],ctdNearestIndex[i][0], ctdNearestIndex[i][1]] \
             for j in range(len(ctdLayer[i]))]
    modTemp.append(t)
    modTemp = np.array(modTemp)
    return modTemp
def array_2dto1d(arr):
    arg = []
    for i in arr:
        for j in i:
            arg.append(j)
    arg = np.array(arg)
    return arg    
FONTSIZE = 25
ctdData = pd.read_csv('ctd_good.csv',index_col=0)
tf_index = np.where(ctdData['TF'].notnull())[0]
ctdLon, ctdLat = ctdData['LON'][tf_index], ctdData['LAT'][tf_index]
ctdTime = pd.Series(np_datetime(ctd['END_DATE'][tf_index], index = tf_index))
ctdTemp = pd.Series(str2ndlist(ctdData['TEMP_VALS'][tf_index]),index=tf_index)
ctdDepth = pd.Series(str2ndlist(ctdData['TEMP_DBAR'][tf_index]), index=tf_index)
ctdMaxDepth = ctdData['MAX_DBAR'][tf_index]
ctdLayer = pd.Series(str2ndlist(ctdData['modDepthLayer'][tf_index],bracket=True), index=tf_index)
ctdNearestIndex = pd.Series(str2ndlist(ctdData['modNearestIndex'][tf_index], bracket=True), index=tf_index)

starttime = datetime(2009, 8, 24)
endtime = datetime(2013, 12, 13)
tempObj = wtm.waterCTD()
url = tempObj.get_url(starttime, endtime)
# modTemp = np.array(tempObj.watertemp(ctdLon.values, ctdLat.values, ctdDepth))
modData = tempObj.get_data(url)
oceantime = modData.variables['oceantime']
modTempAll = modData.variables['temp']
modTemp = pd.Series(getModTemp(modTempAll, ctdTime, ctdLayer, ctdNearestIndex), index=tf_index)

obsTempOff, obsTempOn = [], []
modTempOff, modTempOn = [], []
i = ctdMaxDepth.values>50

obsTempOff = ctdTemp[i]
modTempOff = modTemp[i]

obsTempOn = ctdTemp[~i]
modTempOn = modTemp[~i]
indOff, obsTOff, modTOff, layerOff = [],[],[],[]
for i in obsTempOff.index:
    for j in range(len(obsTempOff[i])):
        y = modTempOff[i][j]
        x = obsTempOff[i][j]
        if y > x + 10:
            indOff.append(i)
            obsTOff.append(x)
            modTOff.append(y)
            layerOff.append(ctdLayer[i][j])
dataOff = pd.DataFrame({'lon': ctdLon[indOff].values,
                        'lat': ctdLat[indOff].values,
                        'time': ctdTime[indOff].values,
                        'obstemp': np.array(obsTOff),
                        'modtemp': np.array(modTOff),
                        'layer': np.array(dep)
                        })
indOn, obsTOn, modTOn, layerOn=[],[],[],[]
for i in obsTempOn.index:
    for j in range(len(obsTempOn[i])):
        y = modTempOn[i][j]
        x = obsTempOn[i][j]
        if y > x + 10:
            indOn.append(i)
            obsTOn.append(x)
            modTOn.append(y)
            layerOn.append(ctdLayer[i][j])
dataOn = pd.DataFrame({'lon': ctdLon[indOn].values,
                       'lat': ctdLat[indOn].values,
                       'time': ctdTime[indon].values,
                       'obstemp': np.array(obsTOn),
                       'modtemp': np.array(modTOn),
                       'layer': np.array(dep)
                       })
time1 = datetime(2013, 07,10)
time2 = starttime + timedelta(hours=1)
layer = 4

tempObj = wtm.water_roms()
