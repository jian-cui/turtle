'''
Author: Jian Cui
'''
import numpy as np

def str2list(s, bracket=False):
    '''
    a is a string converted from a list
    a = '[3,5,6,7,8]'
    b = str2list(a, bracket=True)
    or
    a = '3,4,5,6'
    b = str2list(a)
    '''
    if symb:
        s = s[1:-1]
    s = s.split(',')
    s = [float(i) for i in s]
    return s
def str2ndlist(arg, bracket=False):
    ret = []
    for i in arg:
        a = str2list(i, bracket=bracket)
        ret.append(a)
    ret = np.array(ret)
    return ret
