import os
trial_name = os.path.splitext(__file__)[0]
model_filename = os.path.sep.join(["output", trial_name,"model.h5"])
plot_png_filename = os.path.sep.join(["output", trial_name,"plot.png"])
state_json_filename = os.path.sep.join(["output", trial_name,"state.json"])
checkpoint_folder = os.path.sep.join(["output", trial_name])

from pathlib import Path
Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


from keras.models import Sequential,load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv2DTranspose, BatchNormalization, UpSampling2D, Reshape, Dropout
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def decouple(df):
    matrix = {}
    classes = {}
    for index, row in df.iterrows():
        if row['filename'] not in matrix:
            matrix[row['filename']] = {'count':0}
        if row['class'] not in matrix[row['filename']]:
            matrix[row['filename']][row['class']] = 1
        else:
            matrix[row['filename']][row['class']] += 1
        matrix[row['filename']]['count'] += 1

        if row['class'] not in classes:
            classes[row['class']] = 1
        else:
            classes[row['class']] += 1
    #classes = {k: v for k, v in sorted(classes.items(), key=lambda item: item[1])}            
    return matrix,classes

df_train =  pd.read_csv('data/train_labels.csv')
df_test =  pd.read_csv('data/test_labels.csv')

matrix_train,classes_train = decouple(df_train)
matrix_test,classes_test = decouple(df_test)

train_datagen = ImageDataGenerator(
        rescale=1./255,validation_split=0.2)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory='data/train',
        x_col='filename',
        y_col='class',
        target_size=(440, 440),
        batch_size=1,
        class_mode='categorical',
        subset="training",)

validation_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory='data/train',
        x_col='filename',
        y_col='class',
        target_size=(440, 440),
        batch_size=1,
        class_mode='categorical',
        subset="validation",)


if os.path.isfile(model_filename):
    model = load_model(model_filename)
else:
    model = Sequential()

    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), padding='same', input_shape = (440, 440, 3), activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)) # antes era 0.25
    # Adding a second convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)) # antes era 0.25
    # Adding a third convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)) # antes era 0.25
    # Step 3 - Flattening
    model.add(Flatten())
    # Step 4 - Full connection
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(units = 40, activation = 'softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save(model_filename)

# construct the set of callbacks
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
callbacks = [
	EpochCheckpoint(checkpoint_folder, every=1,startAt=0),
	TrainingMonitor(plot_png_filename,jsonPath=state_json_filename,startAt=0)
]

history = model.fit(
        train_generator,
        #steps_per_epoch=100,
        epochs=1,
        #steps_per_epoch=100,
        validation_data=validation_generator,
        #validation_steps=100
        callbacks=callbacks,
        verbose=1
        )
model.save(model_filename)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
