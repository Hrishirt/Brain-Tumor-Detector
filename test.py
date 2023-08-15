import cv2 
from keras.models import load_model
from PIL import Image
import numpy as np


model = load_model('10epochTumordetectorCategorical.h5')

predictionImage = cv2.imread('/Users/hrishishah/Desktop/Brain Tumor Detection 2/pred/pred5.jpg')

pred = Image.fromarray(predictionImage)

pred = pred.resize((64,64))

pred = np.array(pred)

input_image = np.expand_dims(pred, axis=0)

# Binary System
# prediction = (model.predict(input_image) > 0.5).astype("int32")

# Multi class system
prediction = np.argmax(model.predict(input_image), axis=-1)



print(prediction)