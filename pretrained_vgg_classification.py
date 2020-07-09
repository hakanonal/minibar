import os
trial_name = os.path.splitext(__file__)[0]
model_filename = os.path.sep.join(["output", trial_name,"model.h5"])
plot_png_filename = os.path.sep.join(["output", trial_name,"plot.png"])
state_json_filename = os.path.sep.join(["output", trial_name,"state.json"])
checkpoint_folder = os.path.sep.join(["output", trial_name])

config = {
    'learning_rate':0.0001,
    'epoch':100,
    'batch_size':4,
    'initial_epoch': 0,
    'model_name': 'VGG16_classifier_multi',
    'input_shape_height' : 375,
    'input_shape_width' : 262,
    'continue_training': False
}
#os.environ['WANDB_MODE'] = 'dryrun'
#os.environ["WANDB_RESUME"] = "must"
#os.environ["WANDB_RUN_ID"] = "2xw6r2is"

from pathlib import Path
Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)

#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


from keras.models import Sequential,load_model
#from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv2DTranspose, BatchNormalization, UpSampling2D, Reshape, Dropout
from keras.layers import Dense, Flatten
#from keras import backend as K
#from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    _ = tf.config.experimental.set_memory_growth(physical_devices[0], True)

import wandb
from wandb.keras import WandbCallback
if(config is None):
    wandb.init(project="minibar")
    config = wandb.config
else:
    wandb.init(project="minibar",config=config)

def decouple(df):
    matrix = {}
    classes = {}
    for _, row in df.iterrows():
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
        validation_split=0.2,horizontal_flip=True)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory='data/train',
        x_col='filename',
        y_col='class',
        target_size=(config['input_shape_height'], config['input_shape_width']),
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset="training",)

validation_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory='data/train',
        x_col='filename',
        y_col='class',
        target_size=(config['input_shape_height'], config['input_shape_width']),
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset="validation",)


if os.path.isfile(model_filename) and config['continue_training']:
    model = load_model(model_filename)
else:
    model = Sequential()

    model.add(VGG16(include_top=False,input_shape=(config['input_shape_height'], config['input_shape_width'],3)))

    model.add(Flatten())
    model.add(Dense(40,activation="sigmoid"))

    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])
    model.save(model_filename)

# construct the set of callbacks
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
callbacks = [
	EpochCheckpoint(checkpoint_folder, every=1,startAt=0),
    WandbCallback(save_model=False)
]

history = model.fit(
        train_generator,
        #steps_per_epoch=100,
        epochs=config['epoch'],
        #steps_per_epoch=100,
        validation_data=validation_generator,
        #validation_steps=100
        callbacks=callbacks,
        verbose=1,
        initial_epoch=config['initial_epoch']
        )
model.save(model_filename)
