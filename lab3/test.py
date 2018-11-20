from keras import layers
from keras import models
from keras import optimizers

#####MODEL#####
#pick a model
model = models.Sequential()

#####LAYERS#####
#add layers to the model
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()
#####COMPILE#####
#compile your model- specify how your model should learn
#example parameters: metrics, loss, optimizer, etc.
#check cheatsheet for appropriate types
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#Version2
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

#IF DATA IS IMAGE TYPE- USE IMAGE PREPROCESSING
from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
#train_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator(rescale=1./255)

#train_generator = train_datagen.flow_from_directory(
#        # This is the target directory
#        train_dir,
#        # All images will be resized to 150x150
#        target_size=(150, 150),
#        batch_size=20,
#        # Since we use binary_crossentropy loss, we need binary labels
#        class_mode='binary')

#validation_generator = test_datagen.flow_from_directory(
#        validation_dir,
#        target_size=(150, 150),
#        batch_size=20,
#        class_mode='binary')

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
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

