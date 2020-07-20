import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

_config = {
    'batch_size':1,
    'input_shape_height' : 375,
    'input_shape_width' : 262,
    'model_filename' : 'output/simple_cnn_classification/model.h5'
}

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

#df_train =  pd.read_csv('data/train_labels.csv')
df_test =  pd.read_csv('data/test_labels.csv')

from helpers.decouple import decouple
matrix_test,_ = decouple(df_test)
from helpers.matrix_to_df import matrix_to_df
df_test_agg = matrix_to_df(matrix_test)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test_agg,
        directory='data/test',
        x_col='filename',
        y_col='class',
        target_size=(_config['input_shape_height'], _config['input_shape_width']),
        batch_size=_config['batch_size'],
        class_mode='categorical',
        subset="training",)

model = load_model(_config['model_filename'])
score = model.evaluate_generator(test_generator)
print(score)
