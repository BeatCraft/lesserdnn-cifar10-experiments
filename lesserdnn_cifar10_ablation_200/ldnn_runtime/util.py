#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os
import sys
import time
from stat import *
#import random
#import copy
#import struct
import pickle
import math
import numpy as np
import csv

from PIL import Image
from PIL import ImageFile
from PIL import JpegImagePlugin
from PIL import ImageFile
from PIL import PngImagePlugin
import zlib

# LDNN
#import core


#
# constant values
#
C_PURPLE = (255, 0, 255)
C_RED = (255, 0, 0)
C_ORANGE = (255, 128, 0)
C_YELLOW = (255, 255, 0)
C_RIGHT_GREEN = (128, 255, 128)
C_GREEN = (0, 255, 0)
C_RIGHT_BLUE = (128, 128, 255)
C_BLUE = (0, 0, 255)
COLOR_PALLET = [C_BLUE, C_RIGHT_BLUE, C_GREEN, C_RIGHT_GREEN, C_YELLOW, C_ORANGE, C_RED, C_PURPLE]
#
#
#
def cross_emtropy_error(y, y_len, t, t_len):
    if y_len != t_len:
        return None
    #
    s = 0.0
    for i in range(y_len):
        s += (np.log(y[i])*t[i])
    #
    s = s * -1.0
    return s

def cross_emtropy_error_2(y, y_len, t, t_len):
    delta = 1e-7
    #print np.log(y + delta)
    return -np.sum(t * np.log(y + delta))

#def cross_emtropy_error_fast(y, t, t_class):
#    return np.log(y[t_class]) * t[t_class] * -1
#

def cross_emtropy_error_fast(y, t_class):
    delta = 1e-7
    return -np.log(y[t_class]+delta)

def mean_squared_error_np(y, y_len, t, t_len):
    return np.sum( (y-t)**2.0 )/float(y_len)

def mean_squared_error(y, y_len, t, t_len):
    if y_len != t_len:
        return None
    #
    s = 0.0
    for i in range(y_len):
        s += (y[i]-t[i])**2.0
    #
    s = s / float(y_len)
    return s

def mean_squared_error_B(y, y_len, t, t_len):
    if y_len != t_len:
        return None
    #
    s = 0.0
    for i in range(y_len):
        s += abs(y[i]-t[i])
    #
    return s
    
#
# mostly, mean_absolute_error is used
#
def mean_absolute_error(y, y_len, t, t_len):
    if y_len != t_len:
        return None
    #
    s = 0.0
    for i in range(y_len):
        s += abs(y[i]-t[i])
    #
    return s/float(y_len)

def img2List(img):
    pix = img.load()
    w = img.size[0]
    h = img.size[1]
    #
    ret = []
    for y in range(h):
        for x in range(w):
            k = pix[x, y]
            ret.append(k)
        #
    #
    return ret

def loadData(path):
    img = Image.open(path)
    img = img.convert("L")
    #img = img.resize((14,14))
    data = img2List(img)
    return data

def pickle_save(path, data):
    with open(path, mode='wb') as f:
        pickle.dump(data, f)

def pickle_load(path):
    try:
        with open(path, mode='rb') as f:
            data = pickle.load(f)
            return data
        #
    except:
        return None
    #
#
#
#
def exportPng(r, num_of_processed):
    connections = r.getConnections()
    wlist = []
    for con in connections:
        w = con.getWeightIndex()
        wlist.append(w)
        #
        lc = r.countLayers()
        bc = lc - 1
        width = 0
        height = 0
    #
    for i in range(bc):
        left = r.getLayerAt(i)
        left_nc = left.countNodes()
        right = r.getLayerAt(i+1)
        right_nc = right.countNodes()
        width = width + right_nc*4
        if left_nc + right_nc > height:
            height = left_nc*4 + right_nc*4
        #
    #
    img = Image.new("RGB", (width+100, height+10))
    pix = img.load()
    windex = 0
    w = 0
    h = 0
    #
    for i in range(bc):
        left = r.getLayerAt(i)
        left_nc = left.countNodes()
        right = r.getLayerAt(i+1)
        right_nc = right.countNodes()
        #
        start_w = left_nc*4*i + 10*i
        start_h = 0
        #
        for x in range(right_nc):
            w = start_w + x*4
            for y in range(left_nc):
                h = start_h + y*4
                #print "[%d]%d, %d : %d" % (i, w, h, windex)
                wv = wlist[windex]
                v = COLOR_PALLET[wv]
                pix[w,   h  ] = v
                pix[w,   h+1] = v
                pix[w,   h+2] = v
                pix[w+1, h  ] = v
                pix[w+1, h+1] = v
                pix[w+1, h+2] = v
                pix[w+2, h  ] = v
                pix[w+2, h+1] = v
                pix[w+2, h+2] = v
                windex = windex + 1
            #
        #
    #
    save_name = "./%05d.png" % (num_of_processed)
    img.save(save_name)

def narray2Image(p_1d, x, y):
    print(x, y)
    image = np.zeros(x*y*3, dtype='uint8')
    print(image.shape)
    image = image.reshape([y, x, 3])
    print(image.shape)
    for h in range(y):
        for w in range(x):
            image[h][w][0] = p_1d[y*h+w]
            image[h][w][1] = p_1d[y*x+y*h+w]
            image[h][w][2] = p_1d[y*x*2+y*h+w]
        #
    #
    return Image.fromarray(image)
    #return Image.fromarray(np.uint8(image))

def list_to_csv(path, data_list):
    with open(path, 'wb') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(data_list)
    #
    
def csv_to_list(path):
    print("csv_to_list()")
    data_list = []
    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    data_list.append(cell)
                #
            #
        #
        return data_list
    except:
        return data_list
    #
    return data_list

#def get_key_input(prompt):
#    try:
#        c = eval(input(prompt))
#    except:
#        c = -1
#    return c

def get_key_input(prompt):
    c = -1
    try:
        if sys.version_info[0]==2:
            c = input(prompt) # Python2.7
        else:
            c = eval(input(prompt)) # Python3.x
        #
    except:
        c = -1
    #
    return c

def check_weight_distribution(path):
    return 0
    
def echo(data):
    print(data)
    
def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argc)
    #
    print("test")

if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
