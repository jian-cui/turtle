Turtle Project
==============
###Datafile:
######1. ctd_extract_good.csv_(ctd_extract_TF.py)_:
  - **TF**, get good ctd data, If TF==True, good; If False, bad.

######2. ctd_good.csv_(nearestIndexInMod.py)_:
  - **TF**, get good ctd data, If TF==True, good; If False, bad.
  - **modNearestIndex**, return the index of nearest point in model.
  - **modDepthLayer**, return whcih layer in model observation belongs to.

######3. ctdWithModTempByDepth.csv_(ctdWithModTempByDepth.py)_:
  - **TF**, get good ctd data, If TF==True, good; If False, bad.
  - **modNearestIndex**, return the index of nearest point in model.
  - **modDepthLayer**, return whcih layer in model observation belongs to.
  - **modTempByDepth**, return the temp in model calculated by depth rather than layer.

---
###Module:
######1. turtleModule.py
  - **mon_alpha2num** Return num from name of month
  - **np_datetime** Return a datetime from ctd observation "END_DATE"
  - **bottom_value** Return the bottom temp from obs "TEMP_VALS" str
  - **index_by_depth** Return a list with 2 part divided by 'depth'
  - **str2list** Convert a str to list
  - **str2ndlist** Convert a str to multidimensional arrays(especially for new column added to datafile)
  - **angle_conversion**
  - **dist** Calculate the dist from longitude and latitude
  - **closet_num** Return the index of the closet number in list
  - **draw_basemap** Draw basemap
  - **intersection** Calculate point of intersection of 2 lines

######2. watertempModule.py
    *Note: Using module named [jata](https://github.com/jian-cui/moj/blob/master/jata.py)*
  - This is a module of classes we might use.

---
###Code:
######1. ctd_extract_TF.py
  - Create new data file "ctd_extract_good.csv" with new column *TF*.(For every ctd position, if it has at least one gps position within 3km and 3h, it's good.)

######2. nearestIndexInMod.py
  - Create new data file "ctd_good.csv" with new column *TF*, *modNearestIndex*, *modDepthLayer*

######3. ctdWithModTempByDepth.py
  - Create new data file "ctdWithModTempByDepth.csv" with new column *TF*, *modNearestIndex*, *modDepthLayer*, *modTempByDepth*

######4. dataMap.py:
  - Draw data map of "raw_ctd", "good_ctd", "raw_gps", "good_gps" and so on.

######5. errorMapLayer.py
  - **errorMapLayer4In1.png** Plot 4 maps in 1 fig to show which layer has the most errors
  - **errorMapLayerBar.png** Error bar
  - **errorMapLayerDepthBar.png** Error depth bar

######6. errorMapDepth.py
  - **errorMapDepth4In1.png** Plot 4 maps in 1 fig to show which depth has the most errors
  - **errorMapDepthErrorBar.png** Error bar
  - **errorMapDepthRatioOfError.png**

######7. obsVSmodel_bottomtemp.py
  - Draw the correlation of the deepest observation(we assume it's the bottom of ocean) and appropriate model data.

######8. obsVSmodel_deepestbottom.py
  - If the deepest observation depth is “>50m”(or “<50m”, or “all”), draw the correlation of this observation and appropriate model data.

######9. obsVSmodel_deepshallow.py
  - Draw the correlation of observation and model between deep and shallow(50m)

######10. obsVSmodel_shore.py
  - Draw the correlation of observation and model between onshore and offshore(50m)

######11. deepestDepth.py
  - Return ratio of the deepest depth

######12. timeSeries.py
  - Draw temp change of one specific turtle data and model data.

######13. gridOfError.py
  - Divide the whole area into drifferent girds, and plot the number of observation and error in each grid.