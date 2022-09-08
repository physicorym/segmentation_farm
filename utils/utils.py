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