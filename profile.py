import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import netCDF4
import sys
sys.path.append('../moj')
import jata
import matplotlib.pyplot as plt
from matplotlib import path
class water(object):
    def __init__(self, startpoint):
        '''
        get startpoint of water, and the location of datafile.
        startpoint = [25,45]
        '''
        self.startpoint = startpoint
    def get_data(self, url):
        pass
    def bbox2ij(self, lons, lats, bbox):
        """
        Return tuple of indices of points that are completely covered by the 
        specific boundary box.
        i = bbox2ij(lon,lat,bbox)
        lons,lats = 2D arrays (list) that are the target of the subset, type: np.ndarray
        bbox = list containing the bounding box: [lon_min, lon_max, lat_min, lat_max]
    
        Example
        -------  
        >>> i0,i1,j0,j1 = bbox2ij(lat_rho,lon_rho,[-71, -63., 39., 46])
        >>> h_subset = nc.variables['h'][j0:j1,i0:i1]       
        """
        bbox = np.array(bbox)
        mypath = np.array([bbox[[0,1,1,0]],bbox[[2,2,3,3]]]).T
        p = path.Path(mypath)
        points = np.vstack((lons.flatten(),lats.flatten())).T
        tshape = np.shape(lons)
        # inside = p.contains_points(points).reshape((n,m))
        inside = []
        for i in range(len(points)):
            inside.append(p.contains_point(points[i]))
        inside = np.array(inside, dtype=bool).reshape(tshape)
        # ii,jj = np.meshgrid(xrange(m),xrange(n))
        index = np.where(inside==True)
        if not index[0].tolist():          # bbox covers no area
            raise Exception('no points in this area')
        else:
            # points_covered = [point[index[i]] for i in range(len(index))]
            # for i in range(len(index)):
                # p.append(point[index[i])
            # i0,i1,j0,j1 = min(index[1]),max(index[1]),min(index[0]),max(index[0])
            return index
    def nearest_point_index(self, lon, lat, lons, lats, length=(1, 1),num=4):
        '''
        Return the index of the nearest rho point.
        lon, lat: the coordinate of start point, float
        lats, lons: the coordinate of points to be calculated.
        length: the boundary box.
        '''
        bbox = [lon-length[0], lon+length[0], lat-length[1], lat+length[1]]
        # i0, i1, j0, j1 = self.bbox2ij(lons, lats, bbox)
        # lon_covered = lons[j0:j1+1, i0:i1+1]
        # lat_covered = lats[j0:j1+1, i0:i1+1]
        # temp = np.arange((j1+1-j0)*(i1+1-i0)).reshape((j1+1-j0, i1+1-i0))
        # cp = np.cos(lat_covered*np.pi/180.)
        # dx=(lon-lon_covered)*cp
        # dy=lat-lat_covered
        # dist=dx*dx+dy*dy
        # i=np.argmin(dist)
        # # index = np.argwhere(temp=np.argmin(dist))
        # index = np.where(temp==i)
        # min_dist=np.sqrt(dist[index])
        # return index[0]+j0, index[1]+i0
        index = self.bbox2ij(lons, lats, bbox)
        lon_covered = lons[index]
        lat_covered = lats[index]
        # if len(lat_covered) < num:
            # raise ValueError('not enough points in the bbox')
        # lon_covered = np.array([lons[i] for i in index])
        # lat_covered = np.array([lats[i] for i in index])
        cp = np.cos(lat_covered*np.pi/180.)
        dx = (lon-lon_covered)*cp
        dy = lat-lat_covered
        dist = dx*dx+dy*dy
        
        # get several nearest points
        dist_sort = np.sort(dist)[0:9]
        findex = np.where(dist==dist_sort[0])
        lists = [[]] * len(findex)
        for i in range(len(findex)):
            lists[i] = findex[i]
        if num > 1:
            for j in range(1,num):
                t = np.where(dist==dist_sort[j])
                for i in range(len(findex)):
                     lists[i] = np.append(lists[i], t[i])
        indx = [i[lists] for i in index]
        return indx, dist_sort[0:num]
        '''
        # for only one point returned
        mindist = np.argmin(dist)
        indx = [i[mindist] for i in index]
        return indx, dist[mindist]
        '''
    def nearest_point_index2(self, lon, lat, lons, lats):
        d = dist(lon, lat, lons ,lats)
        min_dist = np.min(d)
        index = np.where(d==min_dist)
        return index
class water_roms(water):
    '''
    ####(2009.10.11, 2013.05.19):version1(old) 2009-2013
    ####(2013.05.19, present): version2(new) 2013-present
    (2006.01.01 01:00, 2014.1.1 00:00)
    '''
    def __init__(self):
        pass
        # self.startpoint = lon, lat
        # self.dataloc = self.get_url(starttime)
    def get_url(self, starttime, endtime):
        '''
        get url according to starttime and endtime.
        '''
        '''
        self.starttime = starttime
        self.days = int((endtime-starttime).total_seconds()/60/60/24)+1 # get total days
        time1 = datetime(year=2009,month=10,day=11) # time of url1 that starts from
        time2 = datetime(year=2013,month=5,day=19)  # time of url2 that starts from
        url1 = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2009_da/avg?lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],mask_rho[0:1:81][0:1:129],u[{0}:1:{1}][0:1:35][0:1:81][0:1:128],v[{0}:1:{1}][0:1:35][0:1:80][0:1:129]'
        url2 = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2013_da/avg_Best/ESPRESSO_Real-Time_v2_Averages_Best_Available_best.ncd?mask_rho[0:1:81][0:1:129],u[{0}:1:{1}][0:1:35][0:1:81][0:1:128],v[{0}:1:{1}][0:1:35][0:1:80][0:1:129],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129]'
        if endtime >= time2:
            if starttime >=time2:
                index1 = (starttime - time2).days
                index2 = index1 + self.days
                url = url2.format(index1, index2)
            elif time1 <= starttime < time2:
                url = []
                index1 = (starttime - time1).days
                url.append(url1.format(index1, 1316))
                url.append(url2.format(0, self.days))
        elif time1 <= endtime < time2:
            index1 = (starttime-time1).days
            index2 = index1 + self.days
            url = url1.format(index1, index2)
        return url
        '''
        self.starttime = starttime
        # self.hours = int((endtime-starttime).total_seconds()/60/60) # get total hours
        # time_r = datetime(year=2006,month=1,day=9,hour=1,minute=0)
        url_oceantime = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?ocean_time[0:1:69911]'
        self.oceantime = netCDF4.Dataset(url_oceantime).variables['ocean_time'][:]
        t1 = (starttime - datetime(2006,01,01)).total_seconds()
        t2 = (endtime - datetime(2006,01,01)).total_seconds()
        self.index1 = self.__closest_num(t1, self.oceantime)
        self.index2 = self.__closest_num(t2, self.oceantime)
        print self.index1, self.index2
        # index1 = (starttime - time_r).total_seconds()/60/60
        # index2 = index1 + self.hours
        # url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?h[0:1:81][0:1:129],s_rho[0:1:35],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],mask_rho[0:1:81][0:1:129],u[{0}:1:{1}][0:1:35][0:1:81][0:1:128],v[{0}:1:{1}][0:1:35][0:1:80][0:1:129]'
        url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?s_rho[0:1:35],h[0:1:81][0:1:129],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],temp[{0}:1:{1}][0:1:35][0:1:81][0:1:129]'
        url = url.format(self.index1, self.index2)
        return url
    def __closest_num(self, num, numlist, i=0):
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
            i = self.__closest_num(num, numlist[indx:],
                              i=i+indx)
        elif num < numlist[indx]:
            i = self.__closest_num(num, numlist[0:indx+1], i=i)
        return i
    def get_data(self, url):
        '''
        return the data needed.
        url is from water_roms.get_url(starttime, endtime)
        '''
        data = jata.get_nc_data(url, 'lon_rho', 'lat_rho', 'temp','h','s_rho')
        return data
    def watertemp(self, lon, lat, depth, time, url):
        data = self.get_data(url)
        lons = data['lon_rho'][:]
        lats = data['lat_rho'][:]
        if type(lon) is list or type(lon) is np.ndarray:
            t = []
            for i in range(len(time)):
                print i
                print 'depth: ', depth[i]
                watertemp = self.__watertemp(lon[i], lat[i], lons, lats, depth[i], time[i], data)
                t.append(watertemp)
                '''
                try:
                    print i, lon[i], lat[i], depth[i], time[i]
                    watertemp = self.__watertemp(lon[i], lat[i], lons, lats, depth[i], time[i], data)
                    t.append(watertemp)
                except Exception:
                    t.append(0)
                    continue
                '''
            t = np.array(t)
        else:
            print 'depth: ', depth
            watertemp = self.__watertemp(lon, lat, lons, lats, depth, time, data)
            t = watertemp
        return t
    def __watertemp(self, lon, lat, lons, lats, depth, time, data):
        '''
        return temp
        '''
        index,d = self.nearest_point_index(lon,lat,lons,lats, num=1)
        print index
        depth_layers = data['h'][index[0][0]][index[1][0]]*data['s_rho']
        print "data['s_rho'][:]", data['s_rho'][:]
        print "h", data['h'][index[0][0]][index[1][0]]
        print 'depth_layer', depth_layers
        layer = np.argmin(abs(depth_layers+depth))
        print 'layer: ', layer
        time_index = self.__closest_num((time-datetime(2006,01,01)).total_seconds(),self.oceantime) - \
            self.index1
        temp = data['temp'][time_index, layer, index[0][0], index[1][0]]
        return temp
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
    if type(m) is str:
        year = int(m[5:9])
        month = mon_alpha2num(m[2:5])
        day =  int(m[0:2])
        hour = int(m[10:12])
        minute = int(m[13:15])
        second = int(m[-2:])
        dt = datetime(year,month,day,hour=hour,minute=minute,second=second)
    else:
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
ctddata = pd.read_csv('ctd_good.csv')
shallow = ctddata.ix[22]        # 17, 19
deep = ctddata.ix[18914]

shallowtime = np_datetime(shallow['END_DATE'])
shallowtemp = [float(temp) for temp in shallow['TEMP_VALS'].split(',')]
modelobj = water_roms()
modelurl = modelobj.get_url(shallowtime, shallowtime+timedelta(hours=1))
depth = [int(dep) for dep in shallow['TEMP_DBAR'].split(',')]
modeltemp = []
for dep in depth:
    print dep
    modeltemp.append(modelobj.watertemp(shallow['LON'], shallow['LAT'], dep,
                                            shallowtime, modelurl))
 
deeptime = np_datetime(deep['END_DATE'])
deeptemp = [float(temp) for temp in deep['TEMP_VALS'].split(',')]
modelobj2 = water_roms()
modelurl2 = modelobj2.get_url(deeptime, deeptime+timedelta(hours=1))
depth2 = [int(dep) for dep in deep['TEMP_DBAR'].split(',')]
modeltemp2 = []
print 'start deep'
for dep2 in depth2:
    modeltemp2.append(modelobj2.watertemp(deep['LON'], deep['LAT'], dep2,
                                              deeptime, modelurl2))

fig = plt.figure()
# ax = fig.add_subplot(121)
ax = fig.add_subplot(111)
ax.plot(modeltemp, depth, 'bo-', label='model')
ax.plot(shallowtemp, depth, 'ro-', label='obs')
ax.set_xlim([10, 30])
ax.set_ylim([40, 0])
ax.set_xlabel('Temp')
ax.set_ylabel('Depth')
ax.set_title('%.2f, %.2f, %s' % (shallow['LAT'], shallow['LON'], str(shallowtime)))
ax.legend(loc='lower right')
'''
ax2 = fig.add_subplot(122)
ax2.plot(modeltemp2, depth2, 'bo-', label='model')
ax2.plot(deeptemp, depth2, 'ro-', label='deep')
ax2.set_xlim([10, 30])
ax2.set_ylim([100, 0])
ax2.set_xlabel('Temp')
ax2.set_ylabel('Depth')
ax2.set_title('%.2f, %.2f, %s' % (deep['LAT'], deep['LON'], str(deeptime)))
ax2.legend(loc='lower right')
'''
plt.show()

