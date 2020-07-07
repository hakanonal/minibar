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

We are going to use this repository, project board and kaggle. We are also using wandb to monitor training process. The project's workpsace is [here](https://app.wandb.ai/hakanonal/minibar). Since this is a private project please request access if you want to check it out.

## Some useful commands to copy paste:
```
#Connect to development environment with GPU
$ ssh -i kolayoto_mp.pem hakanonal@192.168.1.44
```
```
#To watch status of the GPU
$ watch -n 0.1 nvidia-smi
```
```
#To execute the python script on background
$ nohup python3 simple_cnn_classification.py &
```

---

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

#### 01.07.2020

- I have made extensive amount of how to code in cuda. It is a good potensionl if we move with YOLO.
- I have started tthe simplified cnn classification on my server. It is on CPU and working very very slowly.

#### 02.07.2020

- My new GTR 2070 GPU has arrived spent time to install cuda and cudunn. Work in progress...
- Seems that tensorflow supports cuda veresion 10.1 max. However I have installed cuda 11. So I will try to install using [this](https://www.tensorflow.org/install/gpu) page

#### 03.07.2020

- Applied first part of the installation script on [this](https://www.tensorflow.org/install/gpu) page, but got some unknown error. To make it clear. I have removed all drivers. including the cuda 10 and 11. And start all over. 
- I got error on intalling cudunn so I have downloaded the appropiate 10.1 version from [this](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#download) page.
- I finnally I have managed to make the GPU work. Although there are bunch of out of memmory warning messages. It still works.
- I have overcome an error with the [this](https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error) article. (could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR)
- So now we are in a position to itarate faster on development environment.
- I am using [this](https://www.pyimagesearch.com/2019/09/23/keras-starting-stopping-and-resuming-training/) article to continue to train a model that has been trained before.
- I have downloaded the code for callbacks to mark the training process to file. I will modify it that it will detect which epoch it has stoped and set the start epoch as last one. Also I will not save the model for every epoch but I will always save the last model with same filename.
- Started a big 100 epoch train on new GPU. Observing...

#### 06.07.2020

- We have completed the base framework of work environment. Now we can explore different hypterparameters and different networks. The initial network has no converges at all.
- Before moving forward again I want to also try wandb application to monitoring the training. Since It would be much easier to report the results.
- So I have integrated the wandb application. I have better tools to monitor training. Here is the [dashboard](https://app.wandb.ai/hakanonal/minibar) for the entire project.


#### 07.06.2020

- Today I have bunmped into [this](https://towardsdatascience.com/step-by-step-guide-to-using-pretrained-models-in-keras-c9097b647b29) article. It simply explains how to use a pretrained model. So I have decided the use VGG16 as sugested in the article. And I have chopped of the top and add my own dense layer to clasify according to our dataset. I am very curious about the results.