# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:58:13 2021

@author: Soham
"""

import cv2
import numpy as np
actions_table = np.zeros([ 400, 300, 3])
actions_table.fill(255)
cv2.imshow('actions_table', actions_table)
cv2.waitKey(0)
cv2.destroyAllWindows()

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


for i in L_actions_text:
    print((L_actions_text.index(i)+5)*40)
    cv2.putText(actions_table, i, (20,(L_actions_text.index(i)+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255),2)

cv2.imwrite('actions_table.png', actions_table)    


actions_table = cv2.imread('actions_table.png')
