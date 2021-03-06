import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import netCDF4
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
        url_oceantime = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/hidden/2006_da/his?ocean_time'
        # url_oceantime = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2013_da/his_Best/ESPRESSO_Real-Time_v2_History_Best_Available_best.ncd?time'
        self.oceantime = netCDF4.Dataset(url_oceantime).variables['ocean_time'][:]    #if url2006, ocean_time.
        t1 = (starttime - datetime(2006,1,1)).total_seconds() # for url2006 it's 2006,01,01; for url2013, it's 2013,05,18, and needed to be devide with 3600
        t2 = (endtime - datetime(2006,1,1)).total_seconds()
        self.index1 = self.closest_num(t1, self.oceantime)
        self.index2 = self.closest_num(t2, self.oceantime)
        # index1 = (starttime - time_r).total_seconds()/60/60
        # index2 = index1 + self.hours
        # url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?h[0:1:81][0:1:129],s_rho[0:1:35],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],mask_rho[0:1:81][0:1:129],u[{0}:1:{1}][0:1:35][0:1:81][0:1:128],v[{0}:1:{1}][0:1:35][0:1:80][0:1:129]'
        url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/hidden/2006_da/his?s_rho[0:1:35],h[0:1:81][0:1:129],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],temp[{0}:1:{1}][0:1:35][0:1:81][0:1:129],ocean_time'
        # url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2013_da/his_Best/ESPRESSO_Real-Time_v2_History_Best_Available_best.ncd?h[0:1:81][0:1:129],s_rho[0:1:35],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],temp[{0}:1:{1}][0:1:35][0:1:81][0:1:129],time'     
        url = url.format(self.index1, self.index2)
        return url
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
    def get_data(self, url):
        '''
        return the data needed.
        url is from water_roms.get_url(starttime, endtime)
        '''
        data = get_nc_data(url, 'lon_rho', 'lat_rho', 'temp','h','s_rho', 'ocean_time')
        return data
    def watertemp(self, lon, lat, depth, time, url):
        data = self.get_data(url)
        lons = data['lon_rho'][:]
        lats = data['lat_rho'][:]
        temp = data['temp']
        t = []
        for i in range(len(time)):
            print i
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
        return t
    def __watertemp(self, lon, lat, lons, lats, depth, time, data):
        '''
        return temp
        '''
        index = self.nearest_point_index2(lon,lat,lons,lats)
        print index
        depth_layers = data['h'][index[0][0]][index[1][0]]*data['s_rho']
        layer = np.argmin(abs(depth_layers+depth)) # Be careful, all depth_layers are negative numbers
        time_index = self.closest_num((time-datetime(2006,1,1)).total_seconds(),self.oceantime) - \
            self.index1
        temp = data['temp'][time_index, layer, index[0][0], index[1][0]]
        return temp
    def layerTemp(self, layer, url):
        '''
        Get the temp of one whole specific layer.
        Only return temperature of the first 'time' index. 
        '''
        data = self.get_data(url)
        # lons, lats = data['lon_rho'][:], data['lat_rho'][:]
        # index = self.nearest_point_index2(lon, lat, lons, lats)
        # depth_layers = data['h'][index[0][0]][index[1][0]] * data['s_rho']
        # print depth_layers
        # layer = np.argmin(abs(depth_layers + depth))
        # layerTemp = data['temp'][0, layer]
        layerTemp = data['temp'][0, -layer]
        # depthRange = [depth_layers[-layer-1], depth_layers[-layer]]
        # depthRange = [depth_layers[layer-1], depth_layers[layer]]
        return layerTemp
    def depthTemp(self, depth, url):
        '''
        Return temp data of whole area in specific depth to draw contour
        '''
        data = self.get_data(url)
        temp = data['temp'][0]
        layerDepth = data['h']
        s_rho = data['s_rho']
        depthTemp = []
        for i in range(82):
            t = []
            for j in range(130):
                print i, j, 'depthTemp'
                locDepth = layerDepth[i,j]  # The depth of this point
                lyrDepth = s_rho * locDepth
                if depth > lyrDepth[-1]: # Obs is shallower than last layer.
                    d = (temp[-2,i,j]-temp[-1,i,j])/(lyrDepth[-2]-lyrDepth[-1]) * \
                        (depth-lyrDepth[-1]) + temp[-1,i,j]
                elif depth < lyrDepth[0]: # Obs is deeper than first layer.
                    d = (temp[1,i,j]-temp[0,i,j])/(lyrDepth[1]-lyrDepth[0]) * \
                        (depth-lyrDepth[0]) + temp[0,i,j]
                else:
                    ind = self.closest_num(depth, lyrDepth)
                    d = (temp[ind,i,j]-temp[ind-1,i,j])/(lyrDepth[ind]-lyrDepth[ind-1]) * \
                        (depth-lyrDepth[ind-1]) + temp[ind-1,i,j]
                t.append(d)
            depthTemp.append(t)
        return np.array(depthTemp)

class waterCTD(water_roms):
    def watertemp(self, lon, lat, depth, time, url):
        data = self.get_data(url)
        lons = data['lon_rho'][:]
        lats = data['lat_rho'][:]
        t = []
        for i in range(len(time)):
            print i, time[i]
            watertemp = self.__watertemp(lon[i], lat[i], lons, lats, depth[i], time[i], data)
            t.append(watertemp)
        return t
    def __watertemp(self, lon, lat, lons, lats, depth, time, data):
        index = self.nearest_point_index2(lon, lat, lons, lats)
        depth_layers = data['h'][index[0][0]][index[1][0]]*data['s_rho']
        t = []
        # depth = depth.split(',')
        time_index = self.closest_num((time-datetime(2006,1,1)).total_seconds(), self.oceantime) -\
                self.index1
        tem = data['temp'][time_index]
        tem[tem.mask] = 10000
        for dep in depth:
            layer = np.argmin(abs(depth_layers + dep))
            temp = tem[layer, index[0][0], index[1][0]]
            t.append(temp)
            # print time, dep, temp
        return t
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
def get_nc_data(url, *args):
    '''
    get specific dataset from url

    *args: dataset name, composed by strings
    ----------------------------------------
    example:
        url = 'http://www.nefsc.noaa.gov/drifter/drift_tcs_2013_1.dat'
        data = get_url_data(url, 'u', 'v')
    '''
    nc = netCDF4.Dataset(url)
    data = {}
    for arg in args:
        try:
            data[arg] = nc.variables[arg]
        except (IndexError, NameError, KeyError):
            print 'Dataset {0} is not found'.format(arg)
    return data