import os

_config = {
    'learning_rate':0.00001,
    'epoch':100,
    'batch_size':4,
    'initial_epoch': 0,
    'model_name': 'simple_cnn_multi_agg',
    'input_shape_height' : 375,
    'input_shape_width' : 262,
    'continue_training': False
}
os.environ['WANDB_MODE'] = 'dryrun'
#os.environ["WANDB_RESUME"] = "must"
#os.environ["WANDB_RUN_ID"] = "2xw6r2is"

def main(config=None):
    trial_name = os.path.splitext(__file__)[0]
    model_filename = os.path.sep.join(["output", trial_name,"model.h5"])
    checkpoint_folder = os.path.sep.join(["output", trial_name])
    from pathlib import Path
    Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)

    #import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    #import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")


    from keras.models import Sequential,load_model
    #from keras.layers import Dense, , Flatten, , Conv2DTranspose, BatchNormalization, UpSampling2D, Reshape
    from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
    #from keras import backend as K
    #from keras.utils import to_categorical
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import Adam
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

    df_train =  pd.read_csv('data/train_labels.csv')
    #df_test =  pd.read_csv('data/test_labels.csv')

    from helpers.decouple import decouple
    matrix_train,_ = decouple(df_train)
    from helpers.matrix_to_df import matrix_to_df
    df_train_agg = matrix_to_df(matrix_train)

    train_datagen = ImageDataGenerator(
            validation_split=0.2,horizontal_flip=True)

    train_generator = train_datagen.flow_from_dataframe(
            dataframe=df_train_agg,
            directory='data/train',
            x_col='filename',
            y_col='class',
            target_size=(config['input_shape_height'], config['input_shape_width']),
            batch_size=config['batch_size'],
            class_mode='categorical',
            subset="training",)

    validation_generator = train_datagen.flow_from_dataframe(
            dataframe=df_train_agg,
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

        # Step 1 - Convolution
        model.add(Conv2D(32, (3, 3), padding='same', input_shape = (config['input_shape_height'], config['input_shape_width'], 3), activation = 'relu'))
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
        model.add(Dense(units = 40, activation = 'sigmoid'))

        model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])
        model.save(model_filename)

    # construct the set of callbacks
    from helpers.epochcheckpoint import EpochCheckpoint
    callbacks = [
        EpochCheckpoint(checkpoint_folder, every=1,startAt=0),
        WandbCallback(save_model=False)
    ]

    model.fit(
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

if __name__ == '__main__':
    main(_config)
