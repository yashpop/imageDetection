# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 21:04:02 2018

@author: Yashwanth
"""
import pandas as pd
import cv2 as cv2
import os

def registerImageName():
    name = input("Welcome to face recognition! \n Please enter the name of image :")
    #audio code print("")
    return name

def saveImage(cv_object,img, path="", grayscale = None, image_name=None,image_num =None):
    #always convert to grayscale for easy pocessing
    
    #if all([image_name,image_num,grayscale]):
    cv_object.imwrite(path+'_'+image_name+'_'+str(image_num)+'.jpg', grayscale=grayscale)
    cv_object.imshow('image',img)
    return cv_object

def imageProcessing(cv):
    
    path = "imagedata/"
    cam = cv.VideoCapture(0)
    cam.set(3,640) #video,width
    cam.set(4,480) #video, height
    image_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    image_name = registerImageName()
    frame_number = 0
    
    while 1:
        ret, img = cam.read()
        #print(ret, img)
        if ret:
            gray = cv.cvtColor(img, 6)

            #for haar code 1.2 or 1.3 helps in less false positives
            images = image_detector.detectMultiScale(gray,1.3,5)
            for x_o,y_o,w,ht in images:
                cv.rectangle(img,(x_o,y_o),(x_o+w,y_o+ht),color = (255,10,10),thickness = 2)
                frame_number += 1
                #cv2 = saveImage(cv2,img, path, grayscale=gray[y_o:y_o+ht,x_o:x_o+w],image_name=image_name,image_num=frame_number)
                cv.imwrite(path+image_name+'_'+str(frame_number)+'.jpg', gray[y_o:y_o+ht,x_o:x_o+w])
            #lets take >30 samples - gaussian distribution of >=30 samples
            if frame_number>=1100:
                print("Thank you for registering the image")
                cam.release()
                cv.destroyAllWindows()
                break;
                
        else:
            continue
        


        
if __name__ =="__main__":
    
    imageProcessing(cv2)