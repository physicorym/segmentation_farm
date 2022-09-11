import numpy as np
import pandas as pd
import imageio
import random
import os
import sys
import yaml
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

from utils.utils import processing_image, decode_segmentation_masks, get_overlay, draw_window
import models

# подключение конфига
config_name = 'test.yml'
CONFIG_INFO = yaml.safe_load(open(f'config/{config_name}', 'rb'))

# загрузка модели
model = load_model(CONFIG_INFO['segmantation_model'], compile = False)
colormap = np.array([[0,0,0], [255,0,0], [0, 0, 255], [0,255,0],[255,255,255]])
COUNT_WIND = 2

#если бинарник
colormap = colormap.astype(np.uint8)

cap = cv2.VideoCapture(CONFIG_INFO['source_video'])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out_video_3.mp4', fourcc, 10.0, (256, 256))
j = 0

# точки для зоны
pts3 = np.array([[5,10],[5,256],[246,30],[246, 10],[10,10]], np.int32)
pts4 = np.array([[10,246],[246,10],[246,246],[10,10]], np.int32)

pts = [pts3, pts4]


xs = ''
while(cap.isOpened()):
      
    ret, frame = cap.read()
    if ret == True:
        
        # подготовка изображения к модели
        img, imag = processing_image(frame, (256, 256))


        # подключение модели
        predictions = model.predict(img)
        predictions = np.squeeze(predictions)

        predictions= (predictions > 0.5).astype(np.uint8)

        # для визуализации 
        prediction_colormap = decode_segmentation_masks(predictions, colormap, 2)
        overlay = cv2.addWeighted(imag, 0.9, prediction_colormap, 0.5, 0.0)
        cv2.polylines(overlay, [pts3], False, (0,255,0), 2)

        
        
        
        mask_orig = np.zeros(predictions.shape, dtype=np.uint8)

        channel_count = 1 #predictions.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count

        # расчет пикелей в окне
        for j ,point in enumerate(pts):
            mask = draw_window(point, mask_orig)
            masked_image = cv2.bitwise_and(mask, mask, mask = predictions)
            print(np.mean(masked_image), f' номер окна - {j}')
            cv2.imshow(f'frame_0{j}', masked_image)


        # отрисовка основного изображения
        cv2.imshow('Frame', overlay[:,:,::-1])
        

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
    else:
        break
       


cap.release()
out.release()   
# Closes all the frames
cv2.destroyAllWindows()