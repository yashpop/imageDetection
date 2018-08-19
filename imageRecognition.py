# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 08:09:34 2018

@author: Yashwanth
"""

import cv2
import numpy as np
import os
import math
import pickle


def model_retrieval(cv):
    
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('training/trainer.yml')
    cascadePath = 'haarcascade_frontalface_default.xml'
    imageCascade = cv.CascadeClassifier(cascadePath);
    return recognizer,imageCascade,cv

def predict(cam,cv):
    
    

    model, im_cas,cv = model_retrieval(cv)
    min_w = 0.1*cam.get(3)
    min_ht = 0.1*cam.get(4)
    with open('model/users.sav','rb') as fr:
        users = pickle.load(fr)
    #print(users)
    while True:
        
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        images = im_cas.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=8,
                minSize = (int(min_w),int(min_ht))
                )
        font = cv2.FONT_HERSHEY_SIMPLEX
        #print(images)
        for image in images:
            #cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
            x,y,w,h = image
            cv.circle(img,(x+162,y+162),(w+h+72)//3,(0,255,0),2)
            Id, conf = model.predict(gray[y:y+h,x:x+w])
            #print(Id)
            '''
            if(Id==0):
                user = 'shekar'
                conf = conf
                   # Id="yash" 
            if Id==1 or Id == 2:
                user = 'yash'
                conf = conf
            '''
            if users[Id]:
                #print(Id)
                user = users[Id] 
            else:
                user="Unknown"
            #'''
            cv.rectangle(img, (x-20,y-90), (x+w+22, y-22), (255,0,0), -1)
            cv.putText(img, users[model.predict(gray[y:y+h,x:x+w])[0]], (x,y-40), font, 2, (255,255,255), 3)
            cv2.putText(img, str(np.round(conf,3))+'%', (x,y+h), 7, 2, (0,0,200), 4)
        
        cv.imshow('image',img) 
        if cv.waitKey(10) & 0xFF==ord('q'):
            break
    
    
if __name__=="__main__":    
    cam = cv2.VideoCapture(0)
    cam.set(3,1640)
    cam.set(4, 1480)
    predict(cam,cv2)
    cam.release()
    
    cv2.destroyAllWindows()