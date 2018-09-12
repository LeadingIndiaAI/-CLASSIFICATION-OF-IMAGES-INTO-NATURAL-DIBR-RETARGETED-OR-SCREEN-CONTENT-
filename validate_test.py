# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:50:33 2018

@author: ROHITH
"""
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   vertical_flip =True,
                                   fill_mode="nearest",
                                   zoom_range=0.3,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   channel_shift_range=0.3,
                                   rotation_range=30)
 
training_set = test_datagen.flow_from_directory('data/validation',
                                                target_size = (300, 300),
                                                batch_size = 30,
                                                class_mode = 'categorical')
model_final = load_model('data/inception_main_100e.h5')
print('     DIBR         NATURAL      RETARGETTED   SCREENSHOTS  ')

training_set.class_indices

X,y = training_set.next()
result = model_final.predict(X)
print(result)