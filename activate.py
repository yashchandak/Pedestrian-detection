# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:29:26 2016

@author: yash
"""

import numpy as np
e  = 2.718281828



def activate(z, derivative = False, fn = 'ReLu' ):
    
    #Sigmoidal activation function    
    if fn == 'Sigmoid':
       
        if derivative:
            return z*(1-z)        
        return 1/(1+e**-z)
    
    #Leaky Relu activation function    
    elif fn == 'ReLu':
        if derivative:
            return np.array([1 if item>0.01 else 0.01 for item in z])
        else:
            return np.array([max(0.01, item) for item in z])
            
    #tanh activation function
    elif fn == 'Tanh':
        if derivative:
            return 1-(z**2)
        else:
            return (1-e**(-2*z))/(1+e**(-2*z))
            
    else:
        print 'ERROR! invlaid function'