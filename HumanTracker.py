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

import cv2
import numpy as np
import time
import classify

cv2.namedWindow('some',2)

def preprocess(cur, prv):
    
    start = time.clock()    
    cur_blur = cv2.GaussianBlur(cur,(5,5),0)            #Blur the current image
    disp = cur.copy()
    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)       #convert image to b/w with three channels(b,g,r)
    diff =cv2.absdiff(cur_blur, prv)                    #calculate the frame differnece 
    diff = diff.astype(np.uint8)                        #convert differnece to unsigned int
    diff = cv2.GaussianBlur(diff,(5,5),0)               #blur the difference
    print 'diff and 2 gaussian blurs time   :   ', (time.clock()-start)
    
    cv2.imshow('diff', diff)
    
    start = time.clock()
    thresh = cv2.threshold(diff,35,255,cv2.THRESH_BINARY)[1]        #Threshold the differnece image
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)       #Apply Open Morphological transform to disconnect multiple objects
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)     
    temp = np.copy(thresh)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    #compute the contours of all moving objects
    print 'Thresh, morph, contour time   :   ', (time.clock()-start)
    
    cv2.imshow('thresh', temp)
    
    for idx, cont in enumerate(contours):
        tag = False
        x,y,w,h = cv2.boundingRect(cont)        #compute the x,y and width,height of the contour 
        if cv2.contourArea(cont) > 1000:        #ignore small motion contours because of noise
            
            if h > 0.9*w and h < 3*w:           #pre-filter based on structural properties
                #eps = w*h*.001
                #x,y,w,h = int(x-eps), int(y-eps), int(w+2*eps), int(h+2*eps)
            
                eps = 0.05                      #5% increase in all dimensions 
                x,y,w,h = int(x-eps*w), int(y-eps*h), int((1+2*eps)*w), int((1+2*eps)*h)                
                ROI = cur[y:y+h, x:x+w]         #extract the moving object
                #ROI = cv2.equalizeHist(cur[y:y+h, x:x+w])   #Equalize the contrast of extraxted region
                
                if ROI != None:
                    if classify.execute(ROI) == 'people':  #chack if the extracted region is tank
                        tag = True
                        #mark the tank with a bounding box
                        cv2.rectangle(disp, (x,y), (x+w, y+h), (255,0,0), 2)
                        cv2.putText(disp, 'Person', (x,y), 0, 0.5, (255,0,0))
                        
 
        
        #if tag == False:
            #cv2.rectangle(disp, (x,y), (x+w, y+h), (100,100,100), 2) 
       


    return  np.concatenate((disp , cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)), axis=1)     #concatenate classification and thresholded results for display



out = cv2.VideoWriter('output_side3.avi',-1, 24.0, (1440,480))         #initialise video output writer

#cap = cv2.VideoCapture("D:/ToDo/DRDO/IRDatabase/videos/split/12.asf")
cap = cv2.VideoCapture("D:/ToDo/pedestrian/dataset/D.avi")

ret, prv = cap.read()                               #read the first frame
prv = cv2.resize(prv, (720, 480))
prv = cv2.cvtColor(prv,cv2.COLOR_BGR2GRAY)          #convert to b/w
prv = cv2.GaussianBlur(prv,(17,17),0)               #apply gaussian blur

#center_y, center_x  = (item//2 for item in prv.shape)
#center_y -= 75

i = 0
while True:
    ret, img = cap.read()                           #read the image from the camera
    
    if ret:        
        start = time.clock()
        img = cv2.resize(img, (720, 480))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #convert the image to B/W 
        print 'read and convert time   :   ', (time.clock()-start)
        
        result = preprocess(img, prv)               #Process the frame based on initial frame

        print 'total time   :   ', (time.clock()-start)
        #out.write(result)                           #write the frame to the output video file
        cv2.imshow('some', result)
    
    else:
        break
    
    if 0xFF & cv2.waitKey(1) == 27:         #Close the program when 'Esc' is pressed
        break

    
cv2.destroyAllWindows()         #Close all the display windows
cap.release()                   #Release the camera
out.release()                   #release the output writer


"""

import cv2
import picamera
import picamera.array
import time

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        while True:
            camera.capture(stream, format='bgr')
            image = stream.array
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # reset the stream before the next capture
            stream.seek(0)
            stream.truncate()
        cv2.destroyAllWindows()



"""