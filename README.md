# Minibar Beverage Detection

## Objective

The objective of this project is to train a neural network that discovers the beverages in a minibar. We have been already obtained a trained network prior from this project. That network has been traineed using faster-R-CNN using tensorflow. Overall accuracy is around %84 and a single detecetion of an image takes about 9-10 seconds. Another aim of this project is to explore new ways to increase the accuracy and speed.

## Methodology

We have already a datatset that has been annonated including the bounding boxes.

We may try the following approached:
- Create a simple CNN image clasifier and see what is the accurcy and speed.
- Augment the images to increase data
- Try different models like Yolo or SSD
- Instead of extracting the objects bounding boxes, Modify the image clasifier that can addttionally tell the number of objects in the image.
- Use the existing model but try to train it from scratch. (No transfer)

## Environment

We are going to use this repository, project board and kaggle. 

## Journal

#### 24.06.2020

- Today we have spent time to understant the data. The data has been uploaded to [kaggle](https://www.kaggle.com/furkanizmirli/urunler).
- We have spent some time to understand data with [this](https://www.kaggle.com/hakanonal/dataset-discovery) notebook

#### 25.06.2020

- Has been worked on creating a simple cnn clasification network via [this](https://www.kaggle.com/hakanonal/simple-cnn-classification) notebook. First train was with 10 episodes, and the acuracy is terrible. ~%7

#### 30.06.2020

- When trying to predict the model I have realized that I do not know how to re-map the label of the perdicted class numner. To figure it out I have checked dataset discovery nottebook and see that we are actually giving different class maps to train and test datasets. 
- So I decided to combine all dataset and make divide them with keras API.
- I have split the train and validation sets via the keras api. However results for 1 epoch is not different. Every iteration is very slow. I have spent time to buy an appropiate GPU.