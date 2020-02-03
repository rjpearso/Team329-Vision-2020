#!/home/pi/Desktop/launcher.sh python
# import the necessary packages

from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import signal
import cv2
import numpy as np
from piVideoStream329v1 import PiVideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import datetime
import math
from math import e
from networktables import NetworkTables as nt
import sys
import os
import select
import socket

areaFactor = 0.8
count = 1
countframe = 0

import smbus #only need this if you want to send info to the arduino
## It works without it.  Just don't uncomment the smbus lines below.
## We used this to turn a pixel read when we saw something
## it was helpful but total extra

##setup network table connection to robot
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0)
print('Network Tables is setup on Pi')

while 1:
    try:
       ip = socket.gethostbyname('roboRIO-329-FRC.local')
       print('Connected to robot')
       break
    except:
        print('Waiting for NWT and Roborio connection')
        time.sleep(.5)
        pass
nt.initialize(server=ip)
sd = nt.getTable("SmartDashboard")
        #print('No Network Tables:  Aye, Aye')
def contour(cts):
    rect = cv2.minAreaRect(cts)
    box = cv2.boxPoints(rect)
    return(np.int0(box),rect,box)
    
def CalcProperties(ce):
    x = (ce[0] + ce[1])
    dist = (.0357)*x**(1.47))
    return(dist)

def OffsetCalcProperties(ce):
    x = (ce[0] + ce[1])/2
    dist = (.0357)*x**(1.47))
    xOffSet = -5
    theta = math.asin(xOffSet/dist)
    dist = dist * math.cos(theta)
    return(dist, theta)

def piAlive(piIsAlive):
    piIsAlive += 1
    if piIsAlive > 5000000:
        piIsAlive = 0
    sd.putNumber('Pi is alive', piIsAlive)
    return (piIsAlive)

dist = 999  #this is a default error number for distance.  Recognize this in Robot code
vs = PiVideoStream().start() # Start taking frames
time.sleep(2.0)
timerstart = time.time()

display= vs.read()
screenText = ''
rightCenter = 999
leftCenter = 999
turnangle = 999
ang = 31.1
centerOffset = 0
a1 = 0
a2 = 0
arb = 10000
ars = 200
r = 0
centerInches = 11.3

while True:
    count += 1
    turnangle = 999
    try:
        r = piAlive(r)
    except:
        pass

    if count > 200: #Only check network connection every 200 Frames
       count = 0
       start_time = time.time()
       try:
           while True:
               ip = socket.gethostbyname('roboRIO-329-FRC.local')
               nt.initialize(server=ip)
               sd = nt.getTable("SmartDashboard")
               #print("Network reconnected")
               break                
       except Exception as ex:
               print(ex)
               turnAng = 999
               time.sleep(1)
               pass
    if True:
        found = False
        process = True
        frame = vs.read() # read the most recent frame
        if np.array_equal(frame,display) == True: #Checks if this is a unique frame
            Duplicate = True
        else:
            display=frame #set the new frame to display so it can check if the next is new

            ###################################################
            ###################################################
            #  HSV  #  Set HSV values.  These work for our color green
            ###################################################
            ###################################################
            low=np.array([50,180,125])
            high=np.array([70,255,255])
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #Change image to HSV
            mask = cv2.inRange(hsv, low, high) #Apply Mask to image so only targets are white all else is black
            #### The following erodes the mask... This is
            #### not ideal as you can just drop the small contours after and erroding is slow
            #mask = cv2.erode(mask, None, iterations=2)
            #mask = cv2.dilate(mask, None, iterations=4)

            ct = None
            
            
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # Find the contours of the mask
            if len(cnts)!=0:  #only runs the following if you see something
                zzzzzz = 4
               
                #############area=cv2.contourArea(contours) ##don't need 2019 
                
                #x,y,w,h = cv2.boundingRect(cnts[ct]) ## This is if you want vertical and horizontal rectangles not good for 2019 but left it for reference
                
                '''if area < ars or area > arb:
                    too_small.append(ct)
                for c in range(1, len(too_small) + 1):#Get Rid of too small too big
                    del cnts[too_small[-c]]
                            #cnts,area = calcProperties(cnts)'''## if you use the above area you can remove contours now
                    #############################################################
            #try:### Comment out when debugging for hopefully obvious reasons also comment except and pass far below
                #############################################################
                too_small = []
                ct = 0
                for cts in cnts:
                    ### Find boundries and draw rectangle on image
                    
                    boxd,rect,box = contour(cts)#### Creates box form contours
                    
                    cv2.drawContours(display,[box.astype(int)],0,(0,0,255),2) #Draws box on image these are the red boxes.  Its what you see
                    
                    ####cv2.imwrite('test.png', display) ### use this to write images to disk ****Be Careful*** At 60-90 fps thats a lot of pictures and disk space
                    #print(rect) #Prints the details of the minAreaRect following this order: ( center (x,y), (width, height), angle of rotation ) ## Note the order of the rect variable

                    ce,wh,angle = cv2.minAreaRect(box)
                    ce = list(ce)
                    ce[0] = round(ce[0],1)
                    #print(ce[0])
                    if (ce[0] - 320) < 0:
                        centerOffset = round(abs(ce[0]-320),1)
                        tryAgain = True
                        #print(str(centerOffset) + " Pixels to the left away from center.")
                    elif (ce[0]-320) > 0:
                        tryAgain = True
                        centerOffset = round((ce[0] - 320),1)
                        ang = ang*-1
                       # print(str(centerOffset) + " Pixels to the right away from center.")
                    elif (elif (ce[0]-320) <= 5 and (ce[0]-320) >= -5:
                        tryAgain = False
                        print("Perfect Angle")

                    dist1 = (ang/320)*centerOffset
                    dist = CalcProperties(ce)
                    #################
                    #### This gets rid of the things that are too small
                    ###############################################
                    area1 = list(rect)
                    
                    
                    area = area1[1][0]*area1[1][1] ## calculate the area of the boxes
                    
                    if countframe >175:
                        #print (area)
                        countframe = 0
                    countframe +=1
                    #time.sleep(2)
                    if area < ars or area > arb:
                        #print('Found small area')
                        too_small.append(ct)
                    ct +=1
                    
                if ct is not None: ### Check to make sure we still have something
                    
                    for c in range(1, len(too_small) + 1):#Get Rid of too small edit to get rid of too big too if you need it
                        del cnts[too_small[-c]]

                
                centerFind =  800 #just assign to a big number so first value replaces it
                
                if len(cnts) > 1 or len(cnts) < 1:
                    #turn 20 degrees to the see 2 then recheck (robot code)
                    if ce[0] >= 320:
                        turnangle=20 # not sure 20 is the right number
                        #tryAgain = True #tell robot code to try again after move
                    else: #if not right its left
                        turnangle=-20
                       # tryAgain = True
                #########################################
                        #More than 1 target
                #########################################
                #These boxes tells you what you are calculating on the red boxes above show what you see

                elif len(cnts) == 1:
                    target = []
                    for c in cnts:
                          target.append([ce,wh,angle,wh[0]*wh[1]])
                            
                            
                          
                        boxd,rect,box = contour(cts)#### Creates box form contours
                        found = True
                        ####################
                        ## you can add below prints back in for testing
                        ####################
                        
                        #print('@@@@@@@@@@@@@@@@@@')
                        if len(target) >= 1:
                            targetCenter = min(target, key=lambda x:abs(x[0][0]-320))
                            targetMaxArea = max(target, key=lambda x:abs(x[3]))

                            dumpR = []
                            b1=0
                            bb = []
                            bbb = 0
                            ccc = 0
                            for bb in target:
                                #print(bb)
                                #print(rightMaxArea)
                                if .75 * targetMaxArea[3] > float(bb[3]):
                                    dumpR.append(b1)
                                b1 +=1
                                #print(dumpR)
                            if dumpR is not None:
                                for ccc in range(1, len(dumpR) + 1):
                                    del target[dumpR[-ccc]]

                if dist is not None and turnangle is not None:
                    screenText = 'dist=' + str(round(dist,1)) + ' Ang=' + str(round(turnangle,1)) #+ ' dist=' + str(round(dist,1))#+ ' a2=' + str(round(a2,1))str(screenText) + ' Ang=' + str(round(turnangle,1)) + ' dist=' + str(round(dist,1))
                    #cv2.drawContours(display,[box.astype(int)],0,(255,0,0),2) #image channel must be 2 draw blue boxes
                    cv2.line(display,(320,10),(320,470),(255,0,0),2) #draw line in middle of screen
                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10,300)
                    fontScale              = 1
                    fontColor              = (255,255,255)
                    lineType               = 2
                    cv2.putText(display, screenText, 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)### Put text on the screen

                    screenText = ''# reset value of screen text
                    #print('')
                
                PiOffsetDist = 0 #put in its distance back from bumper
                dist = PiOffsetDist + dist
                fps = 30
                try:
                    sd.putNumber("Turn Angle", dist1) #Add these back in to send data to Network tables
                    sd.putNumber("Try Again", tryAgain)
                    sd.putNumber('Distance Away',dist)
                    sd.putNumber('FPS',fps)
                except:
                    pass
##### to write data to Arduino to change light pattern if we see it
##           if found:
##                writeI2C(1)
##            else:
##                writeI2C(2)


            
            #cv2.imshow("Mask", mask) ##look at mask image  ############ COMMENT OUT WHEN ON ROBOT !!!!!!!!!!!!!!!!!! Huge speed penalty
            #cv2.imshow("Frame",display) ## look at what it sees ############ COMMENT OUT WHEN ON ROBOT !!!!!!!!!!!!!!!!!!
            #fps = 1.0 / ((time.time() - start_time)/count)
            #print(fps)
            
            
            #Comment back in to see time per frame
            #print('time', time.time() - currentTime)

            key = cv2.waitKey(1) & 0xFF


    else:
        vs.stop()
        #print('Stopping')
        os.system("sudo shutdown -h now")  ### you can use this to shutdown.  But you should use a battery pack


# do a bit of cleanup
cv2.destroyAllWindows()
vs329.PiVideoStream.stop
        
