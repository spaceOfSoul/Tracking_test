import dlib
import cv2
import numpy as np
import os
import detectingEye as dtt
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
        
        _where = ""
        
        if eyePos == 1:
            _where = "left"
        elif eyePos == 2:
            _where = "right"
        elif eyePos == 3:
            _where = "center"
        else:
            _where = "None"
            
        cv2.putText(frame, _where, (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.imshow("justTrackingEye", frame)
        if cv2.waitKey(1) == 27:
            break
    cam.release()
        
if __name__ == "__main__":
    main()
    