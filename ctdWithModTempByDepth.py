'''
Extract new data file from "ctd_good.csv" to "ctdWithModTempByDepth.csv"
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
def getModTemp(modTempAll, ctdTime, ctdLayer, ctdNearestIndex, s_rho, waterDepth, starttime, oceantime):
    '''
    Return model temp based on observation layers or depth
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
FONTSIZE = 25
ctdData = pd.read_csv('ctd_good.csv')
tf_index = np.where(ctdData['TF'].notnull())[0]
ctdLon, ctdLat = ctdData['LON'][tf_index], ctdData['LAT'][tf_index]
ctdTime = pd.Series(np_datetime(ctdData['END_DATE'][tf_index]), index=tf_index)
ctdTemp = pd.Series(str2float(ctdData['TEMP_VALS'][tf_index]), index=tf_index)
# ctdTemp = pd.Series(bottom_value(ctd['TEMP_VALS'][tf_index]), index=tf_index)
ctdDepth = pd.Series(str2float(ctdData['TEMP_DBAR'][tf_index]), index=tf_index)
ctdLayer = pd.Series(str2ndlist(ctdData['modDepthLayer'][tf_index],bracket=True), index=tf_index)
ctdNearestIndex = pd.Series(str2ndlist(ctdData['modNearestIndex'][tf_index], bracket=True), index=tf_index)

starttime = datetime(2009, 8, 24)
endtime = datetime(2013, 12, 13)
tempObj = wtm.waterCTD()
url = tempObj.get_url(starttime, endtime)
# modTemp1 = tempObj.watertemp(ctdLon.values, ctdLat.values, ctdDepth.values, ctdTime.values, url)
modDataAll = tempObj.get_data(url)
oceantime = modDataAll['ocean_time']
modTempAll = modDataAll['temp']
s_rho = modDataAll['s_rho']
waterDepth = modDataAll['h']
modTemp = getModTemp(modTempAll, ctdTime, ctdLayer, ctdNearestIndex, s_rho, waterDepth, starttime, oceantime)
ctdData['modTempByDepth'] = pd.Series(modTemp, index = tf_index)
ctdData.to_csv('ctdWithModTempByDepth.csv')
