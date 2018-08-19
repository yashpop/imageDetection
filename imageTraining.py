# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 22:21:19 2018

@author: Yashwanth
"""

import cv2
import numpy as np
from PIL import Image
import os
import pickle

class TrainImages:
    
    def __init__(self,path=None):
        self.path = path
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
    
    def get_images_labels(self):
        imagePaths = [os.path.join(self.path,im) for im in os.listdir(self.path)]
        #print(imagePaths)
        imageSamples=[]
        names = []
        
        for imagePath in imagePaths:
            #grab the image and convert to grayscale
            img = Image.open(imagePath).convert('L')
            img_vector = np.array(img,'uint8')
            im_name = os.path.split(imagePath)[-1].split('_')[0]
            images = self.detector.detectMultiScale(img_vector)
            
            for x,y,w,h in images:
                imageSamples.append(img_vector[y:y+h,x:x+w])
                names.append(im_name)
                
        return (imageSamples,names)
    
    def train_images(self):
        imageSamples, names = self.get_images_labels()
        unique_names = list(set(names))
        unique_names.sort()
        #print(names)
        #name_ids = []
        name_ids = list(map(lambda x:unique_names.index(x),names))
        #for x,y in enumerate(unique_names):
        #    name_ids.append(x)
            
        #print(np.array(name_ids))
        self.recognizer.train(imageSamples,np.array(name_ids))
        #saving model to yaml file
        self.recognizer.write('training/trainer.yml')
        
        return unique_names
    
    

if __name__=="__main__":
    t = TrainImages(path = 'imagedata')
    unique_names = t.train_images()
    print(unique_names)
    with open('model/users.sav','wb') as fp:
        pickle.dump(unique_names,fp)