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
