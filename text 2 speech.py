# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:38:55 2021

@author: Soham
"""
from gtts import gTTS 
import os  

def text_2_speech_input(text, index):
      
    # Language in which you want to convert 
    language = 'en'
      
    # Passing the text and language to the engine,  
    # here we have marked slow=False. Which tells  
    # the module that the converted audio should  
    # have a high speed 
    myobj = gTTS(text=text, lang=language, slow=False) 
      
    # Saving the converted audio in a mp3 file named 
    # welcome  
    filename = 'text' + str(index) + '.mp3'
    myobj.save(filename) 
      


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
    text_2_speech_input(text=i,index=L_actions_text.index(i))
    
def text_2_speech_output(textfile):
    # Playing the converted file 
    os.system(textfile+'.mp3') 
    


