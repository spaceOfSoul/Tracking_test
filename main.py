import dlib

import cv2
from cv2 import CV_32F

import numpy as np
import argparse
import os
import detecting_sameFaceModel as dtt
import time



def main():
    cam = cv2.VideoCapture(0)
    gaze  = dtt.Detector()
    
    left_time = 0
    right_time = 0
    center_time = 0
    
    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame,1)
        gaze.face_detect(frame)
        
        eyePos = gaze.detect_gaze()
        emotions = gaze.detect_emotion()
        
        _where = ""
        
        if eyePos == 1:
            _where = "left"
        elif eyePos == 2:
            _where = "right"
        elif eyePos == 3:
            _where = "center"
        else:
            _where = "None"
        print(_where)
            
        cv2.putText(frame, _where, (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, emotions, (100,200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.imshow("eyes and emotions", frame)
        if cv2.waitKey(1) == 27:
            break
    cam.release()

def main2():
    cam = cv2.VideoCapture(0)
    gaze  = dtt.Detector()
    
    left_time = 0
    right_time = 0
    center_time = 0
    
    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame,1)
        gaze.face_detect(frame)
        
        eyePos = gaze.detect_gaze()
        emotions = gaze.detect_emotion()
        
        _where = ""
        
        if eyePos == 1:
            _where = "left"
        elif eyePos == 2:
            _where = "right"
        elif eyePos == 3:
            _where = "center"
        else:
            _where = "None"
        print(_where)
            
        cv2.putText(frame, _where, (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, emotions, (100,200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.imshow("eyes and emotions", frame)
        if cv2.waitKey(1) == 27:
            break
    cam.release()



if __name__ == "__main__":
    main()
    