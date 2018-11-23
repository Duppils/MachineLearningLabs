from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
import sys
import os #os.listdir() lists all files in a directory

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#####MODEL#####
#pick a model
model = models.Sequential()

path = open("path.conf", "r").read()

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

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 4215)
validation_features, validation_labels = extract_features(validation_dir, 2529)
test_features, test_labels = extract_features(test_dir, 2547)

train_features = np.reshape(train_features, (4215, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (2529, 4 * 4 * 512))
test_features = np.reshape(test_features, (2547, 4 * 4 * 512))


#####LAYERS#####
#add layers to the model
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

#if True:
#    sys.exit()
#####COMPILE#####
#compile your model- specify how your model should learn
#example parameters: metrics, loss, optimizer, etc.
#check cheatsheet for appropriate types
model.compile(loss='binary_crossentropy', optimizer=optimizers.nadam(lr=1e-4), metrics=['acc'])

# All images will be rescaled by 1./255
#train_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator(rescale=1./255)
#
#train_generator = train_datagen.flow_from_directory( train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
#
#validation_generator = test_datagen.flow_from_directory( validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

#####TRAIN ON DATA#####
#####ALTERNATIVE 1######
#fit your model to batches of training data
#here you specify which data to use and for what
#model.fit(x_train, y_train, epochs=5, batch_size=32)

#####ALTERNATIVE 2#####
#train your model on the training data directly
#model.train_on_batch(x_batch, y_batch)

#####EVALUTATION#####
#gives information on loss and metrics
#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

#####PREDICTION#####
#now the model can be used for predicting on unknown data
#classes = model.predict(x_test, batch_size=128)

#####SAVE#####
#don't forget to save your model after training with
model.save('flower_model.h5')

#####PRINT RESULTS#####
history = model.fit_generator( train_features, train_labels, epochs=3, batch_size=20, validation_data=(validation_features, validation_labels))
