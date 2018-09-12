# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:40:00 2018

@author: sandeep
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
 
# Image dimensions
img_width, img_height = 250, 250
 
"""
   Creates a CNN model
   p: Dropout rate
   input_shape: Shape of input
"""
def create_model(p, input_shape=(32, 32, 3)):
   # Initialising the CNN
   model = Sequential()
   # Convolution + Pooling Layer
   model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   # Convolution + Pooling Layer
   model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   # Convolution + Pooling Layer
   model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   # Convolution + Pooling Layer
   model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
  
   # Flattening
   model.add(Flatten())
   # Fully connection
   model.add(Dense(64, activation='relu'))
   model.add(Dropout(p))
   model.add(Dense(64, activation='relu'))
   model.add(Dense(64, activation='relu'))
   model.add(Dropout(p/2))
   model.add(Dense(4, activation='softmax'))
  
   # Compiling the CNN
   optimizer = Adam(lr=1e-3)
   metrics=['accuracy']
   model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
   return model
"""
   Fitting the CNN to the images.
"""
def run_training(bs=32, epochs=10):
  
   train_datagen = ImageDataGenerator(rescale = 1./255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True)
   test_datagen = ImageDataGenerator(rescale = 1./255)
 
   training_set = train_datagen.flow_from_directory('data/train',
                                                target_size = (img_width, img_height),
                                                batch_size = bs,
                                                class_mode = 'categorical')
                                               
   test_set = test_datagen.flow_from_directory('data/test',
                                           target_size = (img_width, img_height),
                                           batch_size = bs,
                                           class_mode = 'categorical')
                                          
   model = create_model(p=0.6, input_shape=(img_width, img_height, 3))                                 
   model.fit_generator(training_set,
                        steps_per_epoch=400/bs,
                        epochs = epochs,
                        validation_data = test_set,
                        validation_steps = 80/bs)
   model.save_weights('weights.h5')
def main():
   run_training(bs=32, epochs=50)
 
""" Main """
if __name__ == "__main__":
   main()
"""  import numpy as np
   from keras.preprocessing import image
   test_image1 = image.load_img('single_prediction/cim1_3_4.bmp', target_size = (150, 150))
   test_image1 = image.img_to_array(test_image1)
   test_image1 = np.expand_dims(test_image1, axis = 0)
   test_image2 = image.load_img('single_prediction/8.bmp', target_size = (150, 150))
   test_image2 = image.img_to_array(test_image2)
   test_image2 = np.expand_dims(test_image2, axis = 0)
   images=np.vstack([test_image1,test_image2])
   result = model.predict_classes(images)
   print(result) """
  