# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 11:08:14 2021

@author: Soham
"""

import cv2
import dlib
import numpy as np
import time
from gtts import gTTS 
import os  
from win32api import GetSystemMetrics # to get screen dimensions for setting output window coord

# initiating the face recognition library in dlib
detector_face = dlib.get_frontal_face_detector()
    
# initiating the landmarks for dlib
# the landmarks are a set of 68 points, representing several points 
# mainly defining a face, like the eyebrows, the eyes, lips, face
# boundary. Each point represents the perimeter coord for each face feature
predictor_landmark = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    

########################### function for text to speech from stored voices ###########################

def text_2_speech_output(textfile):
    # Playing the converted file 
    os.system(textfile+'.mp3') 
    
########################### defining function for eye blink ###########################

# function to detect blink, take 6 points,
# looks if the length of the EAR from these
# points drops down below certain threshold
# return blink, else open
def euc_dist(v1,v2):
    return ((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)**0.5
    
    
def eye_blink(image,p1,p2,p3,p4,p5,p6,threshold_blink):
    # detect 'eye aspect ratio' EAR
    # EAR = (euc(p2=37,p6=41) + euc(p3=38,p5=40)) / (2 * euc(p1=36,p4=39))
    EAR = (euc_dist(p2,p6) + euc_dist(p3,p5)) / (2 * euc_dist(p1,p4))
    #print(['blink',EAR])
    if EAR <= threshold_blink:
        #print(['blink',EAR])
        cv2.putText(image, 'blink', (100,400), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255),2)
        #print( 1)
        return(1)
        #return('blink')
    else:
       # print('open')
        return(0)

        




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
            
            # detect blink and return 1
            blink = eye_blink(image = image,p1=landmark_36,p2=landmark_37,
                      p3=landmark_38, p4=landmark_39,
                      p5=landmark_40, p6=landmark_41,
                      threshold_blink = 0.25 )
            
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
            #print(min_x_roi, max_x_roi, min_y_roi,max_y_roi )
            cropped = image_copy[min_y_roi : max_y_roi , min_x_roi : max_x_roi ]
            cropped = cv2.resize(cropped, ((max_x_roi - min_x_roi)*5,(max_y_roi - min_y_roi)*5))
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(cropped_gray, 70, 255, cv2.THRESH_BINARY_INV)[1]
            
            height, width = thresh.shape[0], thresh.shape[1]
            
            # seperating the image into 2 halves from middle
            # looking for ratio of white to black pixels
            # on each side
            left_thresh = thresh[0:height, 0: width//2]
            left_thresh_black = height*width - cv2.countNonZero(left_thresh)
            
            right_thresh = thresh[0:height, width//2 : width]
            right_thresh_black = height*width -  cv2.countNonZero(right_thresh)
           
            # ratio of white pixel count left vs right
            ratio_white =  left_thresh_black/right_thresh_black
            print(ratio_white)
            def direction_reg(image, ratio_white):
                
                if ratio_white < 0.85:
                    #print('right',ratio_white)
                    cv2.putText(image, 'right', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255),1)
                    
                    gaze_output = 1 # right
                    time.sleep(1)
                #elif 0.83 <= ratio_white < 1.2 :
                    
                    #print('centre',ratio_white)
                   # cv2.putText(image, 'centre', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255),1)
                elif ratio_white > 1.25:
                    #print('left',ratio_white)
                    cv2.putText(image, 'left', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255),1)
                    gaze_output = -1 # left
                    time.sleep(1)
                else :
                    gaze_output = 0
                return gaze_output
            
            
            # calling the direction gaze every 3 secs after

            
            #gaze_output = direction_reg
            return([image , thresh, blink, direction_reg(image = image, ratio_white = ratio_white)])
    else:
        return([image])


########################### list of actions ###########################

L_actions = ['text0',
             'text1',
             'text2',
             'text3',
             'text4',
             'text5',
             'text6',
             'text7',
             'text8',
             'text9',
             'text10',
             'text11']


L_actions_text = ['hello',
             'hi! I am Soham',
             'What is your name',
             'How are you',
             'ok',
             'thank you',             
             'yes',
             'no',
             'Please',
             'Please help',
             'Need to go to the washroom',
             'Need a glass of water']



########################### test ###########################

########################### running algo ###########################

L_blink_count=[]
action_reg = L_actions[0] # outside while loop of video

cap =   cv2.VideoCapture(0)
while cap.isOpened():
    _, img = cap.read()
    eye_blink_count = 0
    output = main_detection(image=img)
    ########################### left and right gaze targets ###########################
    width = GetSystemMetrics(0)
    x = width//2 - img.shape[1]//2
    y = 0
    #x = width of target window
    #y = height of target window
    left_target = np.zeros([ img.shape[1]//2, x, 3])
    left_target.fill(255)
    cv2.putText(left_target, 'Left target', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255),2)
    cv2.namedWindow('left target')
    cv2.moveWindow('left target', 0, 0)
    cv2.imshow('left target', left_target)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    right_target =np.zeros([ img.shape[1]//2, x, 3])
    right_target.fill(255)
    cv2.putText(right_target, 'Right target', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255),2)
    cv2.namedWindow('right target')
    cv2.moveWindow('right target', width - img.shape[1]//2 ,0)
    cv2.imshow('right target', right_target)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    ########################### closing of setting left-right target ###########################
    
    if len(output) > 1:
        cv2.namedWindow('Eye Track')
        cv2.moveWindow('Eye Track', width//2 - 60, img.shape[0]+30)
        cv2.imshow('Eye Track', output[1])
        eye_blink_count = output[2] + eye_blink_count
        ########################### loading actions table ###########################

        actions_table = cv2.imread('actions_table.png')
        print(eye_blink_count)  
        input = output[3]
        if input == 1:
            #i = input
            action_reg = L_actions[(L_actions.index(action_reg)+ input)%len(L_actions)]
            print(action_reg )
            cv2.putText(output[0], L_actions_text[L_actions.index(action_reg)], (240,300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255),2)
            cv2.rectangle(actions_table, (10,(L_actions.index(action_reg)+1)*30 - 20), ((285,(L_actions.index(action_reg)+1)*30 + 10)), (255,0, 0))
            cv2.imshow('actions_table', actions_table)

        elif input == -1:
            action_reg = L_actions[(L_actions.index(action_reg)+ input)%len(L_actions)]
            print(action_reg )
            cv2.putText(output[0], L_actions_text[L_actions.index(action_reg)], (240,300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255),2)
            cv2.rectangle(actions_table, (10,(L_actions.index(action_reg)+1)*30- 20), ((285,(L_actions.index(action_reg)+1)*30 + 10)), (255,0, 0))
            cv2.imshow('actions_table', actions_table)

        else:
            print('no action registered')
            cv2.rectangle(actions_table, (10,(L_actions.index(action_reg)+1)*30- 20), ((285,(L_actions.index(action_reg)+1)*30 + 10)), (255,0, 0))            
            cv2.imshow('actions_table', actions_table)

        
        
        
        if eye_blink_count == 1:
            L_blink_count.append(eye_blink_count)
            if len(L_blink_count)>= 7:
                print('input registered')
                print('selection =', action_reg)
                
                ################
                text_2_speech_output(textfile=action_reg)
                ################

                cv2.putText(output[0], 'input registered', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255),2)
                
            
        
        else:
            L_blink_count=[]
            
        cv2.namedWindow('main')
        cv2.moveWindow('main', x, y)
        cv2.imshow('main', output[0])
            
        
    else:
        cv2.namedWindow('main')        
        cv2.moveWindow('main', x, y)
        cv2.imshow('main', output[0])
    
    #out.write(output[0])
        
    #cv2.imshow('main',img)
    #cv2.imshow('main',output_eye)    

    
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

cap.release()
cv2.destroyAllWindows() 

