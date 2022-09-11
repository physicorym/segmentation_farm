import tensorflow as tf 
import cv2
import numpy as np


def processing_image(frame, size):

	image = tf.image.convert_image_dtype(frame, tf.float32)
	image = tf.image.resize(image, size)#, method='nearest')
	imag = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

	imag = cv2.resize(imag, size)

	img = np.expand_dims(image, axis = 0)

	return img, imag


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 1.0, colored_mask, 1.0, 0.0)
    return overlay


def draw_window(points, mask):

    mask = cv2.fillPoly(mask, pts=[points], color=(255, 255, 255))

    return mask
