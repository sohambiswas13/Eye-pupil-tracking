# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 11:08:14 2021

@author: Soham
"""

import cv2
import dlib
import face_recognition
import numpy as np

    
# initiating the face recognition library in dlib
detector_face = dlib.get_frontal_face_detector()
    
# initiating the landmarks for dlib
# the landmarks are a set of 68 points, representing several points 
# mainly defining a face, like the eyebrows, the eyes, lips, face
# boundary. Each point represents the perimeter coord for each face feature
predictor_landmark = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    

########################### defining function for eye blink ###########################

# function to detect blink, take 3 points,
# looks if the length of the line from these
# points drops down below certain threshold
# return blink, else open
def eye_blink(image,x1,y1,x2,y2,threshold_blink):
    length = ((x1-x2)**2 + (y1-y2)**2)**0.5
    if length <= threshold_blink:
        print(['blink',length])

        #return('blink')
    #else:
       # print('open')
        #return('open')

        




########################### main function ###########################

# function detecting the face, the edges of eyes,
# creating a horizontal and vertical lines through 
# the eye to detect blinks
def main_detection(image):
    image_copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # recognising faces and looping over the cood in each frame
    faces = detector_face(gray)
    if len(faces) != 0:
        
        for face in faces:
            
            # storing the coord of face
            x, y, w, h = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(image, (x,y), (w,h), (255,0,0))
            
            #storing the land marks from face coord on gray
            landmark_face = predictor_landmark(gray, face)
            
            # take landmark 36 as the corner point of eye
            landmark_36 = (landmark_face.part(36).x, landmark_face.part(36).y)
            # take landmark 39 as the other corner point of eye
            landmark_39 = (landmark_face.part(39).x, landmark_face.part(39).y)
            # horizontal line through 36 and 39
            cv2.line(image, landmark_36, landmark_39, (0, 255, 0), 1)
            
            # take mean of landmark 37 and 38 for upper x coord of eye
            landmark_37 = (landmark_face.part(37).x, landmark_face.part(37).y)
            landmark_38 = (landmark_face.part(38).x, landmark_face.part(38).y)
            # take mean of landmark 40 and 41 for lower x coord of eye
            landmark_41 = (landmark_face.part(41).x, landmark_face.part(41).y)
            landmark_40 = (landmark_face.part(40).x, landmark_face.part(40).y)
            # vertical line through the middle of eye
            cv2.line(image, ( (landmark_37[0]+landmark_38[0])//2 ,landmark_37[1] ),
                                     ( (landmark_40[0]+landmark_41[0])//2 ,landmark_40[1] ), (0, 255, 0), 1 )
            
            # detect blink
            eye_blink(image = image,x1=(landmark_37[0]+landmark_38[0])//2, y1=landmark_37[1] ,
                      x2=(landmark_40[0]+landmark_41[0])//2, y2=landmark_40[1], 
                      threshold_blink = 7 )
            
            ########################### Detecting the eye motion ###########################
            # setting the eye region of interest
            eye_roi = np.array([(landmark_face.part(36).x, landmark_face.part(36).y),
                                (landmark_face.part(37).x, landmark_face.part(37).y),
                                (landmark_face.part(38).x, landmark_face.part(38).y),
                                (landmark_face.part(39).x, landmark_face.part(39).y),
                                (landmark_face.part(40).x, landmark_face.part(40).y),
                                (landmark_face.part(41).x, landmark_face.part(41).y)])
            
            cv2.polylines(image, [eye_roi], True, (255,0,0),1)
            
            # cutting out the roi
            min_x_roi = np.min(eye_roi[:,0])
            max_x_roi = np.max(eye_roi[:,0])
            min_y_roi = np.min(eye_roi[:,1])
            max_y_roi = np.max(eye_roi[:,1])
            print(min_x_roi, max_x_roi, min_y_roi,max_y_roi )
            cropped = image_copy[min_y_roi : max_y_roi , min_x_roi : max_x_roi ]
            cropped = cv2.resize(cropped, ((max_x_roi - min_x_roi)*5,(max_y_roi - min_y_roi)*5))
            return([image , cropped])
    else:
        return([image])



########################### test ###########################

cap =   cv2.VideoCapture(0)#, cv2.CAP_DSHOW)
while cap.isOpened():
    _, img = cap.read()

    output = main_detection(image=img)
    if len(output) == 2:
        cv2.imshow('main', output[0])
        cv2.imshow('cropped', output[1])
    else:
        cv2.imshow('main', output[0])
 
        
    #cv2.imshow('main',output_main)
    #cv2.imshow('main',output_eye)
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

cap.release()
cv2.destroyAllWindows() 

