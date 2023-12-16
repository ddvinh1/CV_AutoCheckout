# classification_utils.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import keras
from tensorflow.keras.preprocessing import image
import os


def warmup_fd(food_model,class_food,warmup_fd_pth):
    img_fd = image.load_img(warmup_fd_pth, target_size=(224, 224))
    img_array1 = image.img_to_array(img_fd)
    img_array1 = np.expand_dims(img_array1, axis=0)
    #img_array1 /= 255.0  # Assuming your model expects input values in the range [0, 1]

    num_warmup_steps = 3  # You can adjust this based on your needs

    for _ in range(num_warmup_steps):
        food_model.predict(img_array1, verbose = 0)
        #predictions1 = food_model.predict(img_array1)
    '''
    predicted1 = np.argmax(predictions1[0])
    print("predicted: ", class_food[predicted1])
    '''
    print("Food model warmed up successfully!")

def warmup_dr(drink_model,class_drink,warmup_fd_pth):
    img_dr = image.load_img(warmup_fd_pth, target_size=(224, 224))
    img_array2 = image.img_to_array(img_dr)
    img_array2 = np.expand_dims(img_array2, axis=0)
    #img_array2 /= 255.0

    num_warmup_steps = 3  # You can adjust this based on your needs

    for _ in range(num_warmup_steps):
        drink_model.predict(img_array2, verbose = 0)
        #predictions2 = drink_model.predict(img_array2)
    '''
    predicted2 = np.argmax(predictions2[0])
    print("predicted: ", class_drink[predicted2])
    '''
    print("Drink model warmed up successfully!")

def classify_image(model, labels, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose = 0)

    pred = np.argmax(prediction[0])
    #print(pred)
    #print(labels[pred])
    return labels[pred]
