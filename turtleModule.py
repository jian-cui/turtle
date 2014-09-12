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
