# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:45:37 2016

@author: yash

For only forward pass during the run-time
"""


import cPickle
import features
import activate as act
import time
import numpy as np


#load the parameters of the neural net
topology = cPickle.load( open("topology.p", "rb"))
act_fn   = cPickle.load( open("act_fn.p", "rb"))
synapses = cPickle.load( open("synapses.p", "rb"))
bias     = cPickle.load( open("bias.p", "rb"))
category = cPickle.load( open("category.p", "rb"))

depth = topology.size - 1
receptors   = [np.zeros(size, 'float') for size in topology[:]]



def execute(inputs):
    global receptors
    
    if inputs.size > 10:
        
        start = time.clock()
        receptors[0] = features.get_HoG(inputs)             #compute the feature vector of the input image
        print 'HoG time   :   ', (time.clock()-start)
        
        start = time.clock()
        for index in xrange(0,depth):                       #Execute Neural Network
            receptors[index+1] = act.activate(synapses[index].dot(receptors[index]) + bias[index+1], False, act_fn[index+1])        
        print 'NNet time   :   ', (time.clock()-start)
        
        
        pos = np.argmax(receptors[depth])                   #Get the position of the maximum value output
        if pos == 1:            
            if category[pos] > 0.9:                         #return 'Tracked' only if more than 90% sure
                return category[pos]
    return category[0]
    
