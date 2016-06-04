# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:05:12 2016

@author: yash

"""

import numpy as np
import os, os.path
import cv2

#define valid file types for images
valid_images = [".jpg",".gif",".png",".tga", ".pgm"]

"""-------------- Manipulation functions ---------------------------"""

def inverted(img, s):
    img = img.astype(float)
    im = cv2.invert(img)            #Invert the image
    return [im]    
    
def flipped(img):    
    im = cv2.flip(img, 1)           #flip the image on the vertical axis
    return [im]    
    
def rename(img, s):    
    return [img]                      #do nothing
        
def lighting(img):
    im1 =  cv2.equalizeHist(img)    #Increase contrast by histogram equalisation
    
    im2 = img.astype(float)    
    l,u = 75, 255                   #define lower and upper values for the new range
    im_bright = im2/255.0*(u-l) + l #stretch the image values to the new range defined above
    im_bright = im_bright.astype(np.uint8)
    
    im_dark = im2*0.4               #Reduce the brightness to 40% of original
    im_dark = im_dark.astype(np.uint8)
    
    return [im1, im_bright, im_bright]


"""----------------------------------------------------------------"""


def manipulate(imgs, fn):
    #iterates over all the images and manipulates according to the function passed
    
    new = []                #intialise the new set of images
    for img in imgs:
        for im in fn(img):
            new.append(im) #append the manipulated image
    return new              #return the entire set of manipulated images


def read_from_folder(path):
    #Reads all the images in the given folder

    imgs = []
    for f in os.listdir(path):              #list all the files in the folder
        ext = os.path.splitext(f)[1]        #get the file extension
        if ext.lower() not in valid_images: #check if the extension is valid for the image
            continue
        filename = path+'/'+f               #create the path of the image
        img = cv2.imread(filename,0)        #read the image
        imgs.append(img)                    #append to the image list
    return imgs



def expand(path, folders):    
    #Data augmentation for the images in 'folders' of the given 'path'

    for idx, folder in enumerate(folders):
        imgs = read_from_folder(path+folder)                #read all the images in the folder  
        count = len(imgs)
        imgs.extend(manipulate(imgs, flipped))              #flip the images and to original list
        imgs.extend(manipulate(imgs, lighting))             #include lighting changes and add to prv list
        
        for i, img in enumerate(imgs[count:]):              #write all the new manipulated images 
            s = str(i)
            cv2.imwrite(path+folder+'/new_'+s+'.jpg', img )
            
    print 'Datset Augmented successfully'


#compile_data("D:/ToDo/DRDO/IRDatabase/dataset/All_faces/train/", ['tank','others'])

