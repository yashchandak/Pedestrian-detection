# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:41:03 2015

@author: yash 


"""

import numpy as np
import os, os.path
import cv2
import features
import morph
import cPickle


valid_images = [".jpg",".png",".tga", ".pgm"]
path = "D:/ToDo/pedestrian/dataset/"
folders = ['other','people']

inp = features.inp
dim = 32
output = len(folders)               
data = []
test = []


def get_data(index):
    return data[index][0], data[index][1] #return the training data and the expected label

def get_test(index):
    
    return test[index][0], test[index][1] #return the test data and the expected label
    
def read_from_folder(path, val):
    #Read all the images in the folder
    imgs = []
    for f in os.listdir(path):              #list all the files in the folder
        ext = os.path.splitext(f)[1]        #get the file extension
        if ext.lower() not in valid_images: #check if the extension is valid for the image
            continue
        filename = path+'/'+f               #create the path of the image
        img = cv2.imread(filename,0)        #read the image
        
        if img == None:                     #Display error if image has nothing
            print 'Error! Blank Image : ' + filename
            continue
        
        feat = features.get_HoG(img)        #Get the features for the image
        imgs.append([feat, val])            #append features with the categorical output
    return imgs
    

def set_feature_parameters():
    features.bins = 8       #number of bins in HoG
    features.inp = inp       #dimension of input vector for neural network
    features.window = 4     #window size = local region for histogram calculation
    features.dim = dim       #image resized dimension
    
    
def compile_data():
    global data,test
    
    morph.expand(path, folders)             #Expand the dataset available
    #set_feature_parameters()  
    
    for idx, folder in enumerate(folders):
        category = np.zeros(output)
        category[idx] = 1                   #categorical output for the images in this folder
        imgs = read_from_folder(path+folder, category)
        count = int(0.98*len(imgs))         #Percentage of images to be trained

        np.random.shuffle(imgs)             #shuffle all the data  
        
        data.extend(imgs[:count])           #add the images from this folder in the data
        test.extend(imgs[count:])           #add the test images from this folder in the test
    
    np.random.shuffle(data)                 #shuffle data again to shuffle folders also
    save()                                  #Save the final data and test sets



def save():
    #Save the generated data so that it can be reused for next training
    print 'Save Data and test'
    cPickle.dump(data, open("data.p", "wb"))
    cPickle.dump(test, open("test.p", "wb"))
    
def load():
    #load the saved data
    global data, test
    data = cPickle.load( open("data.p", "rb"))
    test = cPickle.load( open("test.p", "rb"))
    print 'Dataset loaded'

    
    
#cifar()
load()
#compile_data()
#generate_data()
print 'Datset made successfully'











"""---------------- For testing on Cifar Dataset ---------------------"""

def unpickle(data_file, test_file, label_file):
    
    fo = open(data_file, 'rb')
    data = cPickle.load(fo)
    fo.close()
    
    fo = open(label_file, 'rb')
    label = cPickle.load(fo)
    fo.close()
    
    fo = open(test_file, 'rb')
    test = cPickle.load(fo)
    fo.close()
    
    return (data, test, label)
    

def convert2requiredFormat(data_c):
    data = []
    imgs = data_c.get('data')
    for i in range(len(imgs)):
        val = data_c.get('fine_labels')[i]
        img = data_c.get('data')[i]
        img = img.reshape((3,32,32))
        dst = np.zeros((32,32))        
        for x in range(32):
            for y in range(32):
                dst[x][y] = (img[0][x][y]+img[1][x][y]+img[2][x][y])/3
        
        feat = features.get_HoG(dst)
        
        if val == 85:            
            data.append([feat, [0,1]])
        else:
            data.append([feat, [1,0]])
    return data


def cifar():
    global data, test   
    data_c, test_c, label = unpickle('D:\\ToDo\\datasets\\cifar-100-python\\train', 'D:\\ToDo\\datasets\\cifar-100-python\\test', 'D:\\ToDo\\datasets\\cifar-100-python\\meta')
 
    data = convert2requiredFormat(data_c)
    test = convert2requiredFormat(test_c)

    save()
    
"""--------------------------------------------------------------"""