from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Folder structure

# data/train/RETARGETTED    - RETARGETTED train samples
# data/train/NATURAL        - naturaltrain samples
# data/train/DIBR           - DIBR train samples
# data/train/SCREENSHOTS    - SCREENSHOT train samples


# data/test/RETARGETTED    - RETARGETTED test samples
# data/test/NATURAL        - natural test samples
# data/test/DIBR           - DIBR test samples
# data/test/SCREENSHOTS    - SCREENSHOT test samples


img_width, img_height = 300, 300        # Resolution of inputs
train_data_dir = "data/train"           # Folder of train samples
validation_data_dir = "data/test" # Folder of validation samples
nb_train_samples = 4769                # Number of train samples
nb_validation_samples = 640            # Number of validation samples
batch_size = 16                        # Batch size
epochs = 20                # Maximum number of epochs
# Load INCEPTIONV3
model=applications.VGG16(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
# Freeze first 15 layers
for layer in model.layers[:15]:
	layer.trainable = False
for layer in model.layers[15:]:
   layer.trainable = True
	
# Attach additional layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation="softmax")(x) # 4-way softmax classifier at the end

model_final = Model(input=model.input, output=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

# train data generator (data augmentation)
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True,vertical_flip =True, fill_mode="nearest", zoom_range=0.3, width_shift_range=0.3, height_shift_range=0.3,channel_shift_range=0.3, rotation_range=30)
# test data generator (data augmentation)
test_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True,vertical_flip =True, fill_mode="nearest", zoom_range=0.3, width_shift_range=0.3, height_shift_range=0.3,channel_shift_range=0.3, rotation_range=30)

# load from directory
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode="categorical")
# load from directory
validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), class_mode="categorical")

# save models
checkpoint = ModelCheckpoint("data/vgg_20e.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# early stopping
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# TRAINING
history = model_final.fit_generator(train_generator, samples_per_epoch=nb_train_samples, epochs=epochs, validation_data=validation_generator, nb_val_samples=nb_validation_samples, callbacks=[checkpoint, early])
