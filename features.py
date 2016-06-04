# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:43:46 2015

@author: yash 
"""
import cv2
import numpy as np
import math

dim = 32
window = 4
bins = 8
inp = 400
#inp = ((dim/window)**2)*bins


def get_HoG(img):
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    """
    Temporarily not using HoG to reduce computational time
    If HoG is required, comment the next 4 lines
    """
    img = cv2.resize(img, (20,20))      #resize the image
    img = img.astype(float)             #type cast to float
    img = img/255.0                     #normalise to range 0-1
    return img.reshape(img.size)        #convert and return 1D vector form of the image
    
    
    """
    Get 'Histogram of features' of the image.
    genreates a binary string of size bins*total_window_count
    Each substring of size 'bins' shows the dominating Gradient direction
    The particular position of dominating gradient direction is marked as '1' and rest are '0'
    
    """    
    
    img = cv2.resize(img, (dim+1, dim+1))       #resize to dim + 1 to accomodate for dx and dy 
    img = img.astype(int)                       #type caste to integer
    features2 = np.zeros(inp)                   #initialise the feature vector
    hist = np.zeros(bins)                       #intialise the histogram vector
    index, dx, dy, mag, ang, pos = 0,0,0,0,0,0  #intialise the required variables
    div = 6.28/bins                             #count the division range based on number of bins
    count = 0
    
    for r in xrange(0,dim, window):
        for c in xrange(0,dim, window):
            hist.fill(0) #reset histogram bins
            
            #calculate HoG of the subWindow
            for i in xrange(window):
                for j in xrange(window):
                    dy = img[r+i+1][c+j] - img[r+i][c+j]    #Y gradient
                    dx = img[r+i][c+j+1] - img[r+i][c+j]    #X gradient
                    mag = dx**2 + dy**2                     #Gradient magnitude      
                    ang = math.atan2(dy,dx) + 3.13          #shift range to positive values, i.e. 0 - 6.27
                    pos = int(ang/div)                      #bin position of current direction 
                    hist[pos] += mag                        #accumulate the values in the histogram
            
            #vector of 1 and 0 for gradient direction
            features2[count*bins + np.argmax(hist)] = 1     #Mark the domination graident direction
            count += 1
           
    return features2