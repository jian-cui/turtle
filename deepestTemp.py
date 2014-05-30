import watertempModule as wtm
from  watertempModule import np_datetime, bottom_value
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from datetime import datetime, timedelta

FONTSIZE = 25
ctd = pd.read_csv('ctd_extract_good.csv')
tf_index = np.where(ctd['TF'].notnull())[0]
ctdLon, ctdLat = ctd['LON'][tf_index], ctd['LAT'][tf_index]
ctdTime = pd.Series(np_datetime(ctd['END_DATE'][tf_index]), index=tf_index)
ctdTemp = pd.Series(ctd['END_DATE'][tf_index], index=tf_index)
ctdTemp = pd.Series(bottom_value(ctd['TEMP_VALS'][tf_index]), index=tf_index)
ctdDepth = pd.Series(ctd['TEMP_DBAR'][tf_index], index=tf_index)
# ctdDepth = ctd['MAX_DBAR'][tf_index]

starttime = datetime(2009, 8, 24)
endtime = datetime(2013, 12, 13)
tempObj = wtm.waterCTD()
url = tempObj.get_url(starttime, endtime)
tempT = tempObj.watertemp(ctdLon.values, ctdLat.values, ctdDepth.values, ctdTime.values, url)

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
