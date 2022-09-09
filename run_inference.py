import numpy as np
import pandas as pd
import imageio
import random
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(base_dir)
sys.path.append(base_dir + '/segmentation_farm/')

import tensorflow as tf 
from tensorflow.keras.models import Model, load_model


import matplotlib.pyplot as plt
import cv2
import imageio
from glob import glob

from utils.utils import processing_image, decode_segmentation_masks, get_overlay
import models




path = 'models/models_val006.h5'
#path = 'C:/e/CVFARM/512_loss_Mobile_net_0306_0.7448450922966003_.h5'
model = load_model(path, compile = False)
colormap = np.array([[0,0,0], [255,0,0], [0, 0, 255], [0,255,0],[255,255,255]])

#если бинарник
#colormap = np.array([[255,255,255], [255,255,255], [0, 0, 255], [0,255,0],[255,255,255]])
colormap = colormap.astype(np.uint8)

path_video = 'C:/e/CVFARM/dataset/video/cow_eating_united.mp4' #видео

cap = cv2.VideoCapture(path_video)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out_video_3.mp4', fourcc, 10.0, (256, 256))
j = 0
# Read until video is completed

pts1 = np.array([[130,0],[20,40],[40,60],[170,0], [130,0]], np.int32)
pts2 = np.array([[130,0],[20,40],[40,60],[170,0], [130,0]], np.int32)

pts3 = np.array([[5,10],[10,256],[246,10],[10,10]], np.int32)
pts3 = np.array([[5,10],[5,256],[246,30],[246, 10],[10,10]], np.int32)
pts4 = np.array([[10,246],[246,10],[246,246],[10,10]], np.int32)


xs = ''
while(cap.isOpened()):
      
    ret, frame = cap.read()
    if ret == True:
        
        img, imag = processing_image(frame, (256, 256))


        predictions = model.predict(img)
        predictions = np.squeeze(predictions)

        predictions= (predictions > 0.5).astype(np.uint8)
        prediction_colormap = decode_segmentation_masks(predictions, colormap, 2)
        overlay = cv2.addWeighted(imag, 0.9, prediction_colormap, 0.5, 0.0)
        
        cv2.polylines(overlay, [pts3], False, (0,255,0), 2)

        
        
        
        mask_orig = np.zeros(predictions.shape, dtype=np.uint8)

        channel_count = 1 #predictions.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count
        mask = cv2.fillPoly(mask_orig, pts=[pts3], color=(255, 255, 255))
        

        masked_image_01 = cv2.bitwise_and(mask, mask, mask = predictions)


        print(np.mean(masked_image_01))
        
        out.write(overlay[:,:,::-1])
        cv2.imshow('Frame', overlay[:,:,::-1])
        cv2.imshow('frame', masked_image_01)
        

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
    else:
        break
       

   

cap.release()
out.release()   
# Closes all the frames
cv2.destroyAllWindows()