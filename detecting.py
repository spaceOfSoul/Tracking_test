import dlib
import cv2
import numpy as np
import os
from imutils import face_utils
from keras.models import load_model
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from statistics import mode

class Detector():
    path = os.path.dirname(os.path.abspath(__file__))
    #face
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(path, 'models/shape_68.dat'))
    
    #emotion
    emotion_model_path = './models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')
    emotion_classifier = load_model(emotion_model_path)
    
    emotion_target_size = emotion_classifier.input_shape[1:3]
    emotion_window = []
    
    FRAME_WINDOW = 10
    emotion_offsets = (20, 40)
    
    image = None
    gray = None
    faces = None
    # rgb = None
    
    def face_detect(self, frame):
        global faces, image, gray
        image = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(gray)
    
    def detect_gaze(self):
        global faces,image,gray
         
        for face in faces:
            landmarks = self.predictor(gray, face)
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, image, gray)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, image, gray)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            print(gaze_ratio)
            #0.78
            if gaze_ratio <= 0.8:#left
                return 1
            elif gaze_ratio <= 1.66:
                return 3
                print('center')
            else:
                return 2
    
    
    def detect_emotion(self):
        global faces,image,gray
        for face_coordinates in faces:
            
            x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), self.emotion_offsets)
            gray_face = gray[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (self.emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = self.emotion_classifier.predict(gray_face)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = self.emotion_labels[emotion_label_arg]
            self.emotion_window.append(emotion_text)

            if len(self.emotion_window) > self.FRAME_WINDOW:
                self.emotion_window.pop(0)
            
            return emotion_text
            
def get_gaze_ratio(eye_points, facial_landmarks, frame, gray):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)
    
    cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
    # cv2.imshow("gray", gray)
    

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)

    
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    #
    # cv2.imshow("mask",mask)
    
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    
    
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio
