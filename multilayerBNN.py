# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:02:55 2015

Multilayer backpropagation neural network

[       ]       [       ]       [       ]       [       ]       [       ]
[Input  ]       [Sigmoid]       [sigmoid]       [sigmoid]       [       ]
[vector ] ====> [hidden ] ----> [hidden ] ----> [hidden ] ====> [Output ]
[       ]       [layer 1]       [layer  ]       [layer N]       [       ]
[       ]       [       ]       [  ...  ]       [       ]
 
 ----> full connections


TODO:
1) convert to modular class/object based design
2) [DONE] addition of biases
3) [DONE] generalise to n number of hidden layers
4) *optimisation
        a) momentum [Done]
        b) conjugated gradient descent  
        c) regularisation
5) [DONE]normalise input and output data
6) *simulated annealing - decaying learning parameter/step size
7) [DONE] cache constant intermediate results instead of recalculating
8) end training based on difference from prv error
9) [DONE] Matrix notation for weight updates
10)*intermediate storage of weights in some file (for recovery/comparison) cPickle/CSV
11) geometric image manip
12)GPU usage
13)plotting error function and output for 2d/3d values
14) Add momentum for bias/ generalise bias (if possible)


@author: yash 
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import time
import features
import dataset
import cPickle
import activate as act

"""
NETWORK TOPOLOGY
"""
e           = 2.718281828
inp         = dataset.inp               #input vector dimensions:
nodes_output= dataset.output            #number of outputs

#for Sigmoid
#learning_rate= 0.4
#momentum    = 0.6

#for ReLu and Tanh
learning_rate= 0.005
momentum    = 0.005

iter_no     = 150                      #training iterations

"""
DATA generation, internediate values and weights initialisation
"""
data        = dataset.data              #get data samples
test        = dataset.test              #get test samples
err         = np.zeros(iter_no)         #keep track of sample error after each iteration
test_err    = np.zeros(iter_no)         #keep track of test error after each iteration

#define the topology of the nets
#number of items denote number of layers
#vlaue of each item denote number of nodes in that layer
topology    = np.array([inp,512, 256, nodes_output])


# Define activation function for each layer
#act_fn      = ['Tanh', 'Tanh', 'Tanh', 'Tanh']
act_fn      = ['ReLu', 'ReLu', 'ReLu', 'Sigmoid']
#act_fn      = ['Sigmoid', 'Sigmoid', 'Sigmoid', 'Sigmoid']
depth       = topology.size - 1


#Initialise all the required variables
synapses    = [np.random.randn(size2,size1)*(1.0/np.sqrt(size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
prv_update  = [np.zeros((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
curr_update = [np.zeros((size2,size1)) for size1,size2 in zip(topology[0:depth],topology[1:depth+1])]
bias        = [np.random.random((size))*0.1 for size in topology[:] ]
receptors   = [np.zeros(size, 'float') for size in topology[:]]
deltas      = [np.zeros(size, 'float') for size in topology[:]]



    
def plotit(x,y, fig, xlabel, ylabel, title):
    #For plotting the error graphs

    plt.figure(1)           #figure number
    plt.plot(x, y)          #pass the X and Y values to be plotted
    plt.xlabel(xlabel)      #Label for X axis
    plt.ylabel(ylabel)      #Label for Y axis
    plt.title(title)        #Give the Plot a title
    plt.show()              #Display the graph
    
    
def train_nets():
    global  err, test_err, deltas, synapses, prv_update, curr_update
    
    error = 0    
    for epoch in xrange(iter_no):        
        #update based on each data point
    
        error_sum = 0
        test_error_sum = 0
        
        for i in xrange(len(data)):                 #Train and learn the parameters on the training data
            inputs, expected = dataset.get_data(i)  #get next training data and label
            
            execute_net(inputs)                     #fwd pass of the inputs in the net
            error = expected - receptors[depth]     #error vector corresponding to each output
            error_sum += sum(abs(error))            #Absolute sum of error across all the classes
                     
            #backpropagation using dynamic programming
            deltas[depth] = act.activate(receptors[depth],True, act_fn[depth])*error
            for index in xrange(depth-1, -1, -1):
                deltas[index] = act.activate(receptors[index],True, act_fn[index])*synapses[index].transpose().dot(deltas[index+1])
            
            #update all the weights
            for index in xrange(depth-1, -1, -1):
                curr_update[index]  = deltas[index+1].reshape(topology[index+1],1)*receptors[index]
                synapses[index]     += learning_rate*curr_update[index] + momentum*prv_update[index]
                bias[index+1]       += learning_rate*deltas[index+1]
            
            prv_update = curr_update                #cur_updates become the prv_updates for next data
         
        
        for i in xrange(len(test)):                 #Evaluate the quality of net on validation test set
            inputs, expected = dataset.get_test(i)  #Get the next validation set data and label
            execute_net(inputs)                     #fwd pass of the inputs in the net
            
            tt = np.zeros(nodes_output)
            pos = np.argmax(receptors[depth])
            tt[pos] = 1                             #determine the output class based on highest score
                
            test_error_sum += sum(abs(expected - tt))#calculate total misclassification
        
        err[epoch] = error_sum/len(data)
        test_err[epoch] = test_error_sum/(2*len(test)) #single misclassification creates an error sum of 2.
        
        if epoch%1 == 0:
            print "Iteration no: ", epoch, "    error: ", err[epoch], " test error: ", test_err[epoch]
            
        if np.argmin(err[:epoch+1]) == epoch:       #should be argmin of test_err actually
            save()                                  #Save the values if it's better than all the previous ones

    
def execute_net(inputs):
    #compute one fwd pass of the network
    global synapses, receptors
    
    receptors[0] = inputs
    for index in xrange(0,depth): 
        receptors[index+1] = act.activate(synapses[index].dot(receptors[index]) + bias[index+1], False, act_fn[index+1]) 
     
def predict(img):    
    #predict the class of the input image
    execute_net(features.get_HoG(img))              #Execute net base don the input image
    pos = np.argmax(receptors[depth])               #Find the class having max. weight
    print receptors[depth], dataset.folders[pos]    #Print the folder name corresponding to the class
    return dataset.folder[pos]

def save():
    #Save all the required parameter, so that they can be imported directly during run-time
    
    print 'Saved Parameters'
    cPickle.dump(topology, open("topology.p", "wb"))
    cPickle.dump(act_fn, open("act_fn.p", "wb"))
    cPickle.dump(synapses, open("synapses.p", "wb"))
    cPickle.dump(bias, open("bias.p", "wb"))
    cPickle.dump(dataset.folders, open("category.p", "wb"))

def load():
    #During run-time directly load the trained parameters

    global topology, act_fn, synapses, bias, depth
    topology = cPickle.load( open("topology.p", "rb"))
    act_fn   = cPickle.load( open("act_fn.p", "rb"))
    synapses = cPickle.load( open("synapses.p", "rb"))
    bias     = cPickle.load( open("bias.p", "rb"))
    dataset.folders     = cPickle.load( open("category.p", "rb"))
    depth = topology.size - 1


def main():
    train_nets()
    plotit(range(iter_no), err, 1, 'iteration number', 'error value', 'Error PLot')         #Plot the Training error
    plotit(range(iter_no), test_err, 1, 'iteration number', 'error value', 'Error PLot')    #Plot validation set error

#load()
start = time.clock()   
main()
end = time.clock()
print 'time elapsed: ', end-start