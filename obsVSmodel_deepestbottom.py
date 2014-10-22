'''
If the deepest observation depth is >50m(or <50m, or all), draw the correlation of this observation and appriate model data.
'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import netCDF4
import matplotlib.pyplot as plt
import sys
sys.path.append('../moj')
import jata
import utilities
from matplotlib import path
from scipy import stats
from watertempModule import *
#class water(object):
#    def __init__(self, startpoint):
#        '''
#        get startpoint of water, and the location of datafile.
#        startpoint = [25,45]
#        '''
#        self.startpoint = startpoint
#    def get_data(self, url):
#        pass
#    def bbox2ij(self, lons, lats, bbox):
#        """
#        Return tuple of indices of points that are completely covered by the 
#        specific boundary box.
#        i = bbox2ij(lon,lat,bbox)
#        lons,lats = 2D arrays (list) that are the target of the subset, type: np.ndarray
#        bbox = list containing the bounding box: [lon_min, lon_max, lat_min, lat_max]
#    
#        Example
#        -------  
#        >>> i0,i1,j0,j1 = bbox2ij(lat_rho,lon_rho,[-71, -63., 39., 46])
#        >>> h_subset = nc.variables['h'][j0:j1,i0:i1]       
#        """
#        bbox = np.array(bbox)
#        mypath = np.array([bbox[[0,1,1,0]],bbox[[2,2,3,3]]]).T
#        p = path.Path(mypath)
#        points = np.vstack((lons.flatten(),lats.flatten())).T
#        tshape = np.shape(lons)
#        # inside = p.contains_points(points).reshape((n,m))
#        inside = []
#        for i in range(len(points)):
#            inside.append(p.contains_point(points[i]))
#        inside = np.array(inside, dtype=bool).reshape(tshape)
#        # ii,jj = np.meshgrid(xrange(m),xrange(n))
#        index = np.where(inside==True)
#        if not index[0].tolist():          # bbox covers no area
#            raise Exception('no points in this area')
#        else:
#            # points_covered = [point[index[i]] for i in range(len(index))]
#            # for i in range(len(index)):
#                # p.append(point[index[i])
#            # i0,i1,j0,j1 = min(index[1]),max(index[1]),min(index[0]),max(index[0])
#            return index
#    def nearest_point_index(self, lon, lat, lons, lats, length=(1, 1),num=4):
#        '''
#        Return the index of the nearest rho point.
#        lon, lat: the coordinate of start point, float
#        lats, lons: the coordinate of points to be calculated.
#        length: the boundary box.
#        '''
#        bbox = [lon-length[0], lon+length[0], lat-length[1], lat+length[1]]
#        # i0, i1, j0, j1 = self.bbox2ij(lons, lats, bbox)
#        # lon_covered = lons[j0:j1+1, i0:i1+1]
#        # lat_covered = lats[j0:j1+1, i0:i1+1]
#        # temp = np.arange((j1+1-j0)*(i1+1-i0)).reshape((j1+1-j0, i1+1-i0))
#        # cp = np.cos(lat_covered*np.pi/180.)
#        # dx=(lon-lon_covered)*cp
#        # dy=lat-lat_covered
#        # dist=dx*dx+dy*dy
#        # i=np.argmin(dist)
#        # # index = np.argwhere(temp=np.argmin(dist))
#        # index = np.where(temp==i)
#        # min_dist=np.sqrt(dist[index])
#        # return index[0]+j0, index[1]+i0
#        index = self.bbox2ij(lons, lats, bbox)
#        lon_covered = lons[index]
#        lat_covered = lats[index]
#        # if len(lat_covered) < num:
#            # raise ValueError('not enough points in the bbox')
#        # lon_covered = np.array([lons[i] for i in index])
#        # lat_covered = np.array([lats[i] for i in index])
#        cp = np.cos(lat_covered*np.pi/180.)
#        dx = (lon-lon_covered)*cp
#        dy = lat-lat_covered
#        dist = dx*dx+dy*dy
#        
#        # get several nearest points
#        dist_sort = np.sort(dist)[0:9]
#        findex = np.where(dist==dist_sort[0])
#        lists = [[]] * len(findex)
#        for i in range(len(findex)):
#            lists[i] = findex[i]
#        if num > 1:
#            for j in range(1,num):
#                t = np.where(dist==dist_sort[j])
#                for i in range(len(findex)):
#                     lists[i] = np.append(lists[i], t[i])
#        indx = [i[lists] for i in index]
#        return indx, dist_sort[0:num]
#        '''
#        # for only one point returned
#        mindist = np.argmin(dist)
#        indx = [i[mindist] for i in index]
#        return indx, dist[mindist]
#        '''
#    def nearest_point_index2(self, lon, lat, lons, lats):
#        d = dist(lon, lat, lons ,lats)
#        min_dist = np.min(d)
#        index = np.where(d==min_dist)
#        return index
#class water_roms(water):
#    '''
#    ####(2009.10.11, 2013.05.19):version1(old) 2009-2013
#    ####(2013.05.19, present): version2(new) 2013-present
#    (2006.01.01 01:00, 2014.1.1 00:00)
#    '''
#    def __init__(self):
#        pass
#        # self.startpoint = lon, lat
#        # self.dataloc = self.get_url(starttime)
#    def get_url(self, starttime, endtime):
#        '''
#        get url according to starttime and endtime.
#        '''
#        '''
#        self.starttime = starttime
#        self.days = int((endtime-starttime).total_seconds()/60/60/24)+1 # get total days
#        time1 = datetime(year=2009,month=10,day=11) # time of url1 that starts from
#        time2 = datetime(year=2013,month=5,day=19)  # time of url2 that starts from
#        url1 = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2009_da/avg?lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],mask_rho[0:1:81][0:1:129],u[{0}:1:{1}][0:1:35][0:1:81][0:1:128],v[{0}:1:{1}][0:1:35][0:1:80][0:1:129]'
#        url2 = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2013_da/avg_Best/ESPRESSO_Real-Time_v2_Averages_Best_Available_best.ncd?mask_rho[0:1:81][0:1:129],u[{0}:1:{1}][0:1:35][0:1:81][0:1:128],v[{0}:1:{1}][0:1:35][0:1:80][0:1:129],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129]'
#        if endtime >= time2:
#            if starttime >=time2:
#                index1 = (starttime - time2).days
#                index2 = index1 + self.days
#                url = url2.format(index1, index2)
#            elif time1 <= starttime < time2:
#                url = []
#                index1 = (starttime - time1).days
#                url.append(url1.format(index1, 1316))
#                url.append(url2.format(0, self.days))
#        elif time1 <= endtime < time2:
#            index1 = (starttime-time1).days
#            index2 = index1 + self.days
#            url = url1.format(index1, index2)
#        return url
#        '''
#        self.starttime = starttime
#        # self.hours = int((endtime-starttime).total_seconds()/60/60) # get total hours
#        # time_r = datetime(year=2006,month=1,day=9,hour=1,minute=0)
#        url_oceantime = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?ocean_time[0:1:69911]'
#        self.oceantime = netCDF4.Dataset(url_oceantime).variables['ocean_time'][:]
#        t1 = (starttime - datetime(2006,01,01)).total_seconds()
#        t2 = (endtime - datetime(2006,01,01)).total_seconds()
#        self.index1 = self.__closest_num(t1, self.oceantime)
#        self.index2 = self.__closest_num(t2, self.oceantime)
#        # index1 = (starttime - time_r).total_seconds()/60/60
#        # index2 = index1 + self.hours
#        # url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?h[0:1:81][0:1:129],s_rho[0:1:35],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],mask_rho[0:1:81][0:1:129],u[{0}:1:{1}][0:1:35][0:1:81][0:1:128],v[{0}:1:{1}][0:1:35][0:1:80][0:1:129]'
#        url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?s_rho[0:1:35],h[0:1:81][0:1:129],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],temp[{0}:1:{1}][0:1:35][0:1:81][0:1:129]'
#        url = url.format(self.index1, self.index2)
#        return url
#    def __closest_num(self, num, numlist, i=0):
#        '''
#        Return index of the closest number in the list
#        '''
#        index1, index2 = 0, len(numlist)
#        indx = int(index2/2)
#        if not numlist[0] < num < numlist[-1]:
#            raise Exception('{0} is not in {1}'.format(str(num), str(numlist)))
#        if index2 == 2:
#            l1, l2 = num-numlist[0], numlist[-1]-num
#            if l1 < l2:
#                i = i
#            else:
#                i = i+1
#        elif num == numlist[indx]:
#            i = i + indx
#        elif num > numlist[indx]:
#            i = self.__closest_num(num, numlist[indx:],
#                              i=i+indx)
#        elif num < numlist[indx]:
#            i = self.__closest_num(num, numlist[0:indx+1], i=i)
#        return i
#    def get_data(self, url):
#        '''
#        return the data needed.
#        url is from water_roms.get_url(starttime, endtime)
#        '''
#        data = jata.get_nc_data(url, 'lon_rho', 'lat_rho', 'temp','h','s_rho')
#        return data
#    def watertemp(self, lon, lat, depth, time, url):
#        data = self.get_data(url)
#        lons = data['lon_rho'][:]
#        lats = data['lat_rho'][:]
#        temp = data['temp']
#        t = []
#        for i in range(len(time)):
#            print i
#            watertemp = self.__watertemp(lon[i], lat[i], lons, lats, depth[i], time[i], data)
#            t.append(watertemp)
#            '''
#            try:
#                print i, lon[i], lat[i], depth[i], time[i]
#                watertemp = self.__watertemp(lon[i], lat[i], lons, lats, depth[i], time[i], data)
#                t.append(watertemp)
#            except Exception:
#                t.append(0)
#                continue
#            '''
#        t = np.array(t)
#        return t
#    def __watertemp(self, lon, lat, lons, lats, depth, time, data):
#        '''
#        return temp
#        '''
#        index = self.nearest_point_index2(lon,lat,lons,lats)
#        print index
#        depth_layers = data['h'][index[0][0]][index[1][0]]*data['s_rho']
#        layer = np.argmin(abs(depth_layers+depth)) # Be careful, all depth_layers are negative numbers
#        time_index = self.__closest_num((time-datetime(2006,01,01)).total_seconds(),self.oceantime) - \
#            self.index1
#        temp = data['temp'][time_index, layer, index[0][0], index[1][0]]
#        return temp
def angle_conversion(a):
    a = np.array(a)
    return a/180*np.pi
def dist(lon1, lat1, lon2, lat2):
    R = 6371.004
    lon1, lat1 = angle_conversion(lon1), angle_conversion(lat1)
    lon2, lat2 = angle_conversion(lon2), angle_conversion(lat2)
    l = R*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2)+\
                        np.sin(lat1)*np.sin(lat2))
    return l
def mon_alpha2num(m):
    month = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    if m in month:
        n = month.index(m)
    else:
        raise Exception('Wrong month abbreviation')
    return n+1
def np_datetime(m):
    dt = []
    for i in m:
        year = int(i[5:9])
        month = mon_alpha2num(i[2:5])
        day =  int(i[0:2])
        hour = int(i[10:12])
        minute = int(i[13:15])
        second = int(i[-2:])
        temp = datetime(year,month,day,hour=hour,minute=minute,second=second)
        dt.append(temp)
    dt = np.array(dt)
    return dt
def mean_value(v):
    v_list = []
    for i in v:
        print i, type(i)
        l = i.split(',')
        val = [float(i) for i in l]
        v_mean = np.mean(val)
        v_list.append(v_mean)
    return v_list
def bottom_value(v):
    v_list = []
    for i in v:
        l = i.split(',')
        val = float(l[-1])
        v_list.append(val)
    v_list = np.array(v_list)
    return v_list
def index_lv(v, n):
    '''
    return a dict
    '''
    index = {}
    for i in range(n):
        index[i] = []
    minv = np.min(v)
    maxv = np.max(v)+0.1
    m = (maxv - minv)/float(n)
    '''
    for i in range(n):
        minvv = minv + i*m
        maxvv = minv + (i+1)*m
        j = 0
        for val in v:
            if val>=maxvv and val<minvv:
                index[i].append(j)
    index = np.array(index)
    return index
    '''
    for i in range(len(v)):
        j = int((v.values[i] - minv)/m) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        index[j].append(v.index[i])
    return index
def index_by_depth(v, depth):
    i = {}
    i[0] = v[v<depth].index
    i[1] = v[v>=depth].index
    return i
def show2pic(x1, y1, fontsize):
    FONTSIZE = fontsize
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    x = np.arange(0.0, 30.0, 0.01)
    '''
    for i in range(10):
        # ax.plot(temp[index[i]], ctdtemp[index[i]], '.', color=colors[i], label='{0}'.format(i))
        ax.scatter(temp[index[i]], ctddata['temp'][index[i]], s=50, c=colors[i], label='{0}'.format(i))
    '''
    # ax.scatter(temp[index[0]], ctddata['temp'][index[0]], s=50, c='b', label='<45')
    # ax.scatter(temp[index[1]], ctddata['temp'][index[1]], s=50, c='r', label='>=45')
    ax1.scatter(x1, y1, s=50, c='b')
    ax1.plot(x, x, 'r-', linewidth=2)
    plt.axis([0, 30, 0, 30], fontsize=15)
    plt.xlabel('Model temp', fontsize=FONTSIZE)
    plt.ylabel('CTD temp', fontsize=FONTSIZE)
    i = x1[x1.isnull()==False].index
    fit = np.polyfit(x1[i], y1[i], 1)
    fit_fn = np.poly1d(fit)
    x2, y2 = x1[i], fit_fn(x1[i])
    plt.plot(x2, y2,'y-', linewidth=2)
    gradient, intercept, r_value, p_value, std_err = stats.linregress(y1[i], x1[i])
    r_squared = r_value**2
    # ax1.set_title('R-squard: %.4f' % r_squared, fontsize=FONTSIZE)
    
    fig2 = plt.figure()
    ax2 =  fig2.add_subplot(111)
    nbins = 200
    H, xedges, yedges = np.histogram2d(x1[i], y1[i], bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0, H)
    plt.pcolormesh(xedges, yedges, Hmasked)
    plt.xlabel('Model temp', fontsize=FONTSIZE)
    plt.ylabel('CTD temp', fontsize=FONTSIZE)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts', fontsize=FONTSIZE)
    cbar.ax.set_yticks(fontsize=20)
    plt.axis([0, 30, 0, 30])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(x, x, 'r-', linewidth=2)
    plt.plot(x2, y2, 'y-', linewidth=2)
    # plt.title('R-squard: %.4f' % r_squared, fontsize=FONTSIZE)
    return ax1, ax2, r_squared
#############################MAIN CODE###########################################
FONTSIZE = 25
# ctd = pd.read_csv('ctd_extract_TF.csv')
ctd = pd.read_csv('ctd_good.csv')
tf_index = np.where(ctd['TF'].notnull())[0]
ctdlat, ctdlon = ctd['LAT'][tf_index].values, ctd['LON'][tf_index].values
ctdtime = np_datetime(ctd['END_DATE'][tf_index])
# ctddepth = mean_value(ctd['TEMP_DBAR'][tf_index])
ctddepth = ctd['MAX_DBAR'][tf_index].values
# ctdtemp = mean_value(ctd['TEMP_VALS'][tf_index])
ctdtemp = bottom_value(ctd['TEMP_VALS'][tf_index])
ctddata = pd.DataFrame({'depth':ctddepth, 'temp':ctdtemp, 'lon':ctdlon,
                        'lat':ctdlat, 'time':ctdtime}).sort_index(by='depth')

starttime = datetime(2009, 8, 24)
endtime = datetime(2013,12 ,13)
tempobj = water_roms()
url = tempobj.get_url(starttime, endtime)
# temp = tempobj.watertemp(ctdlon, ctdlat, ctddepth, ctdtime, url)
temp = tempobj.watertemp(ctddata['lon'].values, ctddata['lat'].values,
                         ctddata['depth'].values, ctddata['time'].values, url)
temp = pd.Series(temp, index = ctddata['temp'].index)

index = index_by_depth(ctddata['depth'], 50)
# colors = utilities.uniquecolors(10)
tp='all'
if tp == 'all':
    x1, y1 = temp, ctddata['temp']
    ax1, ax2, r_squared = show2pic(x1, y1, FONTSIZE)
    ax1.set_title('R-squared: %.4f' % r_squared, fontsize=FONTSIZE)
    ax2.set_title('R-squared: %.4f' % r_squared, fontsize=FONTSIZE)
elif tp == '<50':
    x1, y1 = temp[index[0]], ctddata['temp'][index[0]]
    ax1, ax2, r_squared = show2pic(x1, y1, FONTSIZE)
    ax1.set_title('%s, R-squared: %.4f' % (tp, r_squared), fontsize=FONTSIZE)
    ax2.set_title('%s, R-squared: %.4f' % (tp, r_squared), fontsize=FONTSIZE)
elif tp == '>50':
    x1, y1 = temp[index[1]], ctddata['temp'][index[1]]
    ax1, ax2, r_squared = show2pic(x1, y1, FONTSIZE)
    ax1.set_title('%s, R-squared: %.4f' % (tp, r_squared), fontsize=FONTSIZE)
    ax2.set_title('%s, R-squared: %.4f' % (tp, r_squared), fontsize=FONTSIZE)
plt.show()
'''
# Plot Deepest Data Quantity
fig = plt.figure()
ax = fig.add_subplot(111)
y = ctddata['depth'].values
x = np.arange(1, np.amax(y)+1)
bar = np.array([0]*np.amax(y))
for i in y:
    if i in x:
        bar[i-1] = bar[i-1]+1
plt.barh(x, bar)
plt.ylim([250, 0])
plt.ylabel('depth', fontsize=25)
plt.xlabel('Quantity', fontsize=25)
plt.title('Deepest data histogram', fontsize=25)
plt.show()
'''
