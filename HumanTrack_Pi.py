# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 07:42:35 2016

@author: yash

TODO:

[1] : [Done] Compare with the first frame only, no need to update prv, if background is static

[2] : Implement fwd pass in C++ / Cython + python
[3] : [Done] Deploy entire thing on Pi
[4] : Test Tiny Conv Net
[5] : Use the Pi GPU
[6] : Get better dataset

[7] : RECORD motion vector directly http://picamera.readthedocs.org/en/release-1.10/recipes2.html
[8] : Massive dilation and Add distance-transform to separate overlapping segemnts
[9] : remove any motion which is either not on the edges or was not their around the radius of current centre in the prv frame

"""

from __future__ import print_function
import cv2
import classify
import numpy as np
import time
from PiVideoStream import PiVideoStream
from FPS import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
#from display import Display

def preprocess(cur, prv):
    
    start = time.clock()    
    cur_blur = cv2.GaussianBlur(cur,(5,5),0)            #Blur the current image
    disp = cur.copy()
    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)       #convert image to b/w with three channels(b,g,r)
    diff =cv2.absdiff(cur_blur, prv)                    #calculate the frame differnece 
    diff = diff.astype(np.uint8)                        #convert differnece to unsigned int
    diff = cv2.GaussianBlur(diff,(5,5),0)               #blur the difference
    #print 'diff and 2 gaussian blurs time   :   ', (time.clock()-start)
    
    #cv2.imshow('diff', diff)
    
    start = time.clock()
    thresh = cv2.threshold(diff,35,255,cv2.THRESH_BINARY)[1]        #Threshold the differnece image
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)       #Apply Open Morphological transform to disconnect multiple objects
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)     
    temp = np.copy(thresh)
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    #compute the contours of all moving objects
    #print 'Thresh, morph, contour time   :   ', (time.clock()-start)
    
    #cv2.imshow('thresh', temp)
    
    for idx, cont in enumerate(contours):
        tag = False
        x,y,w,h = cv2.boundingRect(cont)        #compute the x,y and width,height of the contour 
        if cv2.contourArea(cont) > 300:        #ignore small motion contours because of noise
            
            if h > 0.9*w and h < 3*w:           #pre-filter based on structural properties
                #eps = w*h*.001
                #x,y,w,h = int(x-eps), int(y-eps), int(w+2*eps), int(h+2*eps)
            
                eps = 0.05                      #5% increase in all dimensions 
                x,y,w,h = int(x-eps*w), int(y-eps*h), int((1+2*eps)*w), int((1+2*eps)*h)                
                ROI = cur[y:y+h, x:x+w]         #extract the moving object
                ROI = cv2.equalizeHist(cur[y:y+h, x:x+w])   #Equalize the contrast of extraxted region
                
                if ROI != None:
                    if classify.execute(ROI) == 'people':  #check if the extracted region is tank
                        tag = True
                        #mark the tank with a bounding box
                        cv2.rectangle(disp, (x,y), (x+w, y+h), (255,0,0), 2)
                        cv2.putText(disp, 'Person', (x,y), 0, 0.5, (255,0,0))
                        
                        

    return  np.concatenate((disp , cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)), axis=1)     #concatenate classification and thresholded results for display








# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

cv2.namedWindow('disp',2)


# created a *threaded *video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from `picamera` module...")
vs = PiVideoStream().start()

#if args["display"] > 0:
#    disp = Display().start()

time.sleep(2.0)

prv = vs.read()
prv = cv2.cvtColor(prv,cv2.COLOR_BGR2GRAY)
prv = cv2.GaussianBlur(prv,(17,17),0)

center_y, center_x  = (item//2 for item in prv.shape)


fps = FPS().start()
 
# loop over some frames...this time using the threaded stream
while fps._numFrames < args["num_frames"]:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	#frame = imutils.resize(frame, width=400)
 
	# check to see if the frame should be displayed to our screen

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = preprocess(img, prv)
        
        if args["display"] > 0:
            #disp.preview(result)
            cv2.imshow('disp', result)
        if 0xFF & cv2.waitKey(1) == 27:         #Close the program when 'Esc' is pressed
            break
 
	# update the FPS counter
	fps.update()
        #print(frame.shape, fps._numFrames)

        # stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
#disp.close()
cv2.destroyAllWindows()
vs.stop()



