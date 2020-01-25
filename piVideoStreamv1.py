# Do not run on neural network
# import the necessary packages
#Use $15 Amazon 5MP camera module
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import numpy as np
import time
#from imutils.video import FPS

#from picamera.array import PiRGBArray
#from picamera import PiCamera
#import argparse
#import imutils
#import time
import cv2
#import datetime
#import sys
#import os

class PiVideoStream:
    
    def __init__(self):
        #print('did i get to init')
        #############################
        ##  Resolution ##############
        ## Seriously if you are going to change any of this call Pearson
        ## The order these are switched off matters!!! dont change the order
        #####################
        resolution = (640,480)#(480,368)
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.iso = 0 #must be 0
        self.camera.awb_mode = "off" #must be off
        self.camera.drc_strength = 'off'#must be off
        self.camera.image_effect = 'none'#must be off
        self.camera.color_effects = None #must be off
        shutspd = 10000 #shutter speed without gigger
        time.sleep(1) #sleep necessary for exposure_mode
        self.camera.exposure_mode='off'#must be off
        self.camera.exposure_compensation = 0 #must be 0
        self.camera_mode = 7
        #############################
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #############################
        #INSIDE VALUE = 11000
        #OUTSIDE VALUE 5000
        #FIELD VALUE 9000?
        #BOILER VALUE 10000?
        #shutspd = 2500 #shutter speed without gigger
        #time.sleep(2) #sleep necessary for exposure_mode
        #self.camera.exposure_mode='off'#must be off
        self.camera.shutter_speed = shutspd# 2500 with gigger Probably the only thing you need to adjust
        #############################
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #############################
        self.camera.awb_gains = 1.9 #0.0-8.0 typical .9 to 1.9 don't adjust it effects color and hsv values
        self.camera.brightness = 27#27  0-=100
        self.camera.contrast = 100 #-100 to 100
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                format="bgr", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False


    def start(self):
        #print('I made it to start')
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        #print('I Started Stream')
        return self

    def update(self):
##        global stream
##        global frame
##        global stopped
##        global rawCapture
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
                # grab the frame from the stream and clear the stream in
                # preparation for the next frame
                self.frame = f.array
                self.rawCapture.truncate(0)

                # if the thread indicator variable is set, stop the thread
                # and resource camera resources
                if self.stopped:
                        self.stream.close()
                        self.rawCapture.close()
                        self.camera.close()
                        return

    def start2(self):
        Thread(target=self.write, args=()).start()
        return self

                    
    def read(self):
        
        #global frame
        return self.frame

    def stop(self):
        #global stopped
        self.stopped = True

