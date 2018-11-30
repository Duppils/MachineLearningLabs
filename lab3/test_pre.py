from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
import sys
from keras.applications import InceptionV3
import os #os.listdir() lists all files in a directory

conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.summary()

path = str(open("path.conf", "r").read()).rstrip('\n')
dataset = os.path.join(path)

train_dir = os.path.join(dataset, 'train')
validation_dir = os.path.join(dataset, 'validation')
test_dir = os.path.join(dataset, 'test')

#IF DATA IS IMAGE TYPE- USE IMAGE PREPROCESSING
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 3, 3, 2048))
    labels = np.zeros(shape=(sample_count, 5))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),  #Scale image to 150x150 pixels
        batch_size=batch_size,
        class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2595)
validation_features, train_labels = extract_features(validation_dir, 865)
test_features, test_labels = extract_features(test_dir, 865)

train_features = np.reshape(train_features, (2595, 3 * 3 * 2048))
validation_features = np.reshape(validation_features, (865, 3 * 3 * 2048))
test_features = np.reshape(test_features, (866, 3 * 3 * 2048))

#####LAYERS#####
model = models.Sequential()
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels, epochs=5, batch_size=20, validation_data=(validation_features, validation_labels))

