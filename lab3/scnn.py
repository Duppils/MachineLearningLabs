from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
import sys
import os #os.listdir() lists all files in a directory

#from keras.applications import VGG16

#conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#####MODEL#####

path = str(open("path.conf", "r").read()).rstrip('\n')

dataset = os.path.join(path)

train_dir = os.path.join(dataset, 'train')
validation_dir = os.path.join(dataset, 'validation')
test_dir = os.path.join(dataset, 'test')

#Version2
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

#IF DATA IS IMAGE TYPE- USE IMAGE PREPROCESSING
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def get_gen(directory, sample_count):
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),  #Scale image to 150x150 pixels
        batch_size=batch_size,
        class_mode='categorical')
    return generator 


#####LAYERS#####
#add layers to the model
#pick a model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.summary()

from keras.utils import to_categorical
#binary = to_categorical(

train_generator=get_gen(train_dir, 30)
validation_generator=get_gen(validation_dir, 30)

#check cheatsheet for appropriate types
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['acc'])

model.save('scnn.h5')

#####PRINT RESULTS#####
history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 5, validation_data = validation_generator, validation_steps = 50)
