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


#### 07.07.2020

- Today I have bunmped into [this](https://towardsdatascience.com/step-by-step-guide-to-using-pretrained-models-in-keras-c9097b647b29) article. It simply explains how to use a pretrained model. So I have decided the use VGG16 as sugested in the article. And I have chopped of the top and add my own dense layer to clasify according to our dataset. I am very curious about the results.
- I have also traied to run this code on a GPU hosted environment like AWS. I have instenciated a Deep Learning AMI on ubuntu 18.04. All drivers and libraries seems to be installed right away. However, when I execute the script it does not recognize the GPU and continues to work on CPU.
- It seems that deep learning AMI's have different versions of cuda installed. So you can switch by using [this](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-base.html) document
- According to [this](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) document, I have decided that g2 instance type is not suitable. So I dedcided the [chepeast](https://aws.amazon.com/tr/ec2/pricing/on-demand/) recomended instance which is g4dn.xlarge
- We have finally sucesfully run a cloud GPU in AWS. I will also use this instance.
- According to pre-trained verrsion the results were even worse so I am making a terrible error here. I will research more on that.

#### 08.07.2020

- Today I am begining to debug why the results are so low. I am trying to add augmentation to increase the data. Horizantal flip feels right. [This](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/) article is helpful to understand the keras augmentation.
- Also I would like to check how keras Imagedata Generator class generates images in desired augmentaion and sizes(shapes). I will discover that on [this](dataset_discovery.ipynb) old notebook.
- In horizantol flip the texts on the products also flips. I write this just for note. I do not what may be the implications of this.
- Spent some time on ubuntu networking. Following articles helped:
    - https://askubuntu.com/questions/972429/how-to-bring-network-up-on-boot
    - https://help.ubuntu.com/community/NetworkConfigurationCommandLine/Automatic
```
$ ip addr
$ sudo ip link set dev enp12s0 down
$ sudo dhclient enp12s0
$ sudo lshw -c network
$ sudo nano /etc/netplan/01-netcfg.yaml 
```    

- Acdientially I have crached my development ubuntu environment. I have to re install it from scratch. There are the atricles I have used during re-installation.
    - install ssh server https://www.cyberciti.biz/faq/ubuntu-linux-install-openssh-server/
    - ssh login via public/private key authorization https://www.cyberciti.biz/tips/ssh-public-key-based-authentication-how-to.html
    - installed git (sudo apt install git)
    - set git username and email https://dirask.com/posts/Git-how-to-set-username-and-email-MDgRQ1?gclid=Cj0KCQjw3ZX4BRDmARIsAFYh7ZL2KaxOFkspIupa1Et1P4b1-Di0dD3h_JaIRjAlwllSK6JDAG8Ju4kaAnaBEALw_wcB 
    - setup wireless adapter.  sudo nano /etc/netplan/01-netcfg.yaml  https://netplan.io/examples Tried to setup wireless but I could not set it up passing. Because I have re-installed entire ubuntu. Install ubuntu on wireless then activate ethernet using the above commands.
    - Install gcc: sudo apt install gcc
    - Install gnupg: sudo apt-get install gnupg
    - Install cuda and cudunn. https://www.tensorflow.org/install/gpu ubuntu 18.04. Aplied the code here but instead of installing the driver first and then cuda, I installed cuda-10-1 rigth away. Also go over [this](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) document
    - Install pip3: sudo apt install python3-pip
    - Install virtualenv: pip3 install virtualenv
    - In project environment install dependecies: pip install -r requirements.txt
    - and we are back again...
    - noticed that in this fresh installation I did not get memory error.


- So today I wanted to test if I can augment horizantally and also increase the image size. every epoch is a lot slower right now.
    - and I have waited 1 epoch to finish took about ~15min. Either augmentation nor image size did not change the result. 
- I am decreasing the input image shape if it is going to speed up. 
    - Well It has speeded up significantlly however I got the memory error again. However it continue to train. The results seems to be same after one epoch. And no change as seen [here](https://app.wandb.ai/hakanonal/minibar/runs/2uhoeblj)

#### 09.07.2020

- So I will try to discover the data as seen by the network especially for the part. I will try to visualize [here](dataset_discovery.ipynb)
    - It suprizes me that the visualization of feature maps is bunch of random pixels. 
- While I was discovering the layers of the VGG16 I started to looked for a multi object detection on a single image classifier. Bumped into [this](https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/) article. He suggest that change the last activation function softmax to sigmoid and loss to binary_crossentropy. So I will make these changes to predefined model and check what will happen.
    - How dummy am I? the last layer of the previous version has no activation function at all.
    - At last we are seeing some improvements: https://app.wandb.ai/hakanonal/minibar/runs/399erczn
    - Although the accuracy graph has tendency to improve it is very slow.

#### 10.07.2020

- Today I am going to start to try with a lower learning rate 1e2 see what happens. [run](https://app.wandb.ai/hakanonal/minibar/runs/181iooqx)
    - we do not have tendecy of increase in accuracy. So I have kill the process. 
- I am starting a run for learning rate 1e3. [run](https://app.wandb.ai/hakanonal/minibar/runs/1s67da9j)
    - this run also is not improving I am killing int at epoch 15
- I will try to set the learining rate to old value. 1e4 and see If I get similar result with the initial one. I am considering to add seed into config if I do not get consistant result. [run](https://app.wandb.ai/hakanonal/minibar/runs/nlrt81ii)
    - Yes 1e4 learning rate is defniatelly can learn.
    - I will wait tilil at least epoch 20 to make sure.
- On next round I will try to even decrease more. [run](https://app.wandb.ai/hakanonal/minibar/runs/ppqmilw5)
    - 1e5 has even improved weel in terms of loss and validatoion acuracy. However, the acurracy a little bit worse. On the other hand we see the gradual improvement, during time.


- Meanwhile today I am trying to access my local development GPU installed computer from outside network. My router does not allow me to port forward so I am installing a intermideite server on AWS to port forward.
    ```
    # creting a public private key pair for the current user.
    $ ssh-keygen -t rsa
    ```
    - To accomplish this task I have bumped into [this](https://superuser.com/questions/595989/ssh-through-a-router-without-port-forwarding) article.
    - In thoery: The computer inside network should ssh with the parameter -R to the intermideite server and the computer outside the network which want to connect to inside computer should ssh with parameter -J to the intermediate server.
    - After my attempts I could not make it work. It has worked once but when I try to repeat the connection I can not re-connect. Here are the command that may be useful in the future.
    ```
    # This command should be executed from the inside computer that wants to be connected from outside. ubuntu@34.244.80.212 is my intermediate server.
    $ ssh -R 8022:localhost:22 ubuntu@<ip_to_jump_server>
    ```
    ```
    # This command should be executed from your computer to connect inside computer via intermediate server.
    $ ssh -J ubuntu@<ip_to_jump_server> hakanonal@localhost -p8022
    ```
    - [This](https://superuser.com/questions/467398/how-do-i-exit-an-ssh-connection) is a useful article how to end a ssh session from a real computer console. Enter->~.
    - [This](https://man.openbsd.org/OpenBSD-current/man5/ssh_config.5#ProxyJump) reference is useful for ssh .config file directives.

#### 13.07.2020

- I will try 1e5 once again on a cloud server right now. [run](https://app.wandb.ai/hakanonal/minibar/runs/h7qeblhi)
    - The run has finished and we see the same pattren. a fix seed seems is not imporrtant as it is.
    - The overall pererformance is still very bad. Evenworse the validattion loss seems to be incresing aftter some epochs insteead of decreasing.
    - And also it is odd that 1e5 has better loss but worse acuracy. What does it mean?

- I have dicded to set-up an [sweep](https://docs.wandb.com/sweeps) in wandb

#### 14.07.2020

- sweep! To do that I need my script to convert function and make sure to call sweep and give as callback parameter.
    - the sweep has started https://app.wandb.ai/hakanonal/minibar/sweeps/slche9g5

- Meanwhile finally I have managed to setup remote connection to my local GPU development environment. Some helpful articles are below:
    - [This](https://www.ssh.com/ssh/tunneling/example) article is for ssh tunnuling options I have used remote port forwarding for my local development environment.
    - [This](https://www.ssh.com/ssh/sshd_config/) article is for sshd config. I have configured GatewayPorts config to yes in jump AWS server.
    - [This](https://medium.com/maverislabs/proxyjump-the-ssh-option-you-probably-never-heard-of-2d7e41d43464) article is for how to config the ~/.ssh/config file so that I can easiy make connection via vscode.
    - [This](https://www.everythingcli.org/ssh-tunnelling-for-fun-and-profit-autossh/) article gives info about autossh. It is needed for local development computer to keep alive the ssh connection to intermediate jump server.
        ```
        autossh -R 8022:localhost:22 ubuntu@54.217.20.203
        ```


#### 15.07.2020

- I have bumped into [this](https://www.javacodemonk.com/difference-between-loss-accuracy-validation-loss-validation-accuracy-in-keras-ff358faa). It says "val_loss starts increasing, val_acc also increases.This could be case of overfitting or diverse probability values in cases where softmax is being used in output layer" which is our currerrnt case. I will try some dropouts.
- I am also aware that the acuracy is way too small from my target. it should be >%90 but it is <%11
- I am also planing to add extra dense layers with relu at bottom of the network.
- I have created a new script with the [run](https://app.wandb.ai/hakanonal/minibar/runs/2lxvb0rk). The network is begening with 0.25 dropout after vgg and then flatten and dropout 0.5 twice after dense layers. hidden ones has relu acitavtion and last one has sigmoid. Current status is way below from the others.le 1e-5. I will wait till to the end of 100 epoch.
    - This run has a sharper acuracy increrase then other network. However the validation acuracy is stayed put. overall all metrics are worse.
- Am I still feeding the data to network wrong? How can make sure the sturture of the data. When I feed in one image I also give the ground truth that there is one object in there. However if there is multiple objects in the picture then continousuly give different ground truth for the same picture multiple times. I should make sure that every sample has different image and the objects in them should be as array or something like that. back to dataset discovery.
    - Seems that I am feeding the data wrong. Networks sees that every sample has only one object in them. So I continouslly confuse my network. 
    - I need to undertand how the image generator works. so that I can summurize the data by different images.
    - [This](https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do) article helps me understand how generrators and yield keyword works.
    - In [this](https://godatadriven.com/blog/keras-multi-label-classification-with-imagedatagenerator/) article it says that if I summarize the dataframe and set the labels as array, flow_from_dataframe will handle it autmaticlly for me. Let's see is it going to work?
        - First! I have creatded the aggragated dataframe and feed it in to image data generator. I get multiple 1s in the output.
    - I am going to refurbish the [simple_cnn_classification.py](simple_cnn_classification.py)
    - Well it is obviously corrected the issue In the first epoch I got 0.48 acuracy right away. So I am deploying it to my GPU and start a new sweep there.

- The new sweep has been created [sweep](https://app.wandb.ai/hakanonal/minibar/sweeps/dwpeqgg0/overview)
    - It is continuned on working but I am not very interested with the results. Hence I know there is much more improvementt area because of the environment difference. Details are below.

- the new version has a bug though. The model save for every epoch change does not seems to be working.
    - sadly! first sweep is way worse than the old version. I hope it is a very wrong learning rate batch hpyerparameter selection. I will wait for other sweeps.
    - Nevertheless after some 10 more runs we have the hyper parameter that the accuracy has seen the 0.2. Which is an improvement but not as expected. When I have tried the code on my local machince CPU I got 0.5 right away. Moreover the loss is still worse then non aggragatetd wrong data.

- Ok this is really wierd. on my local machine I execute the same code and the acuracy start right away arround 0.65 however in GPU machine the acuracy is hanging below 0.05 which is a huge difference.
    - I have tried to run CPU insttead of GPU on GPU enabled divce but nothing changed. My local environment has python 3.7 and GPU enabled device has 3.6. How can we find some other reasons?

#### 16.07.2020

- I will continue to work on environment inconsistancies. I will try to find the initial acuracy arround 0.5 or at least not below 0.1
    - Let's check this version diffeerences. I will check the status on cloud server.
        - Cloud 3.7.7 with GPU negative 0.05
        - Cloud 3.6.10 with GPU positive I got 0.65 initially right away. and even validation acuracy is 0.8 it is even considerablly improving fast.
        - localdev 3.6.9 with GPU negative 0.05
        - local 3.7.2 no GPU positive 0.65
    - So what the h.. is going on here?
    - I am reading [this](https://realpython.com/intro-to-pyenv/) article
    - I have checked [this](https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development) article about keras seed feeding.

#### 17.07.2020

- I will try to install pyenv on my local so that I can compare different versions in the same hardware.
    - [This](https://github.com/pyenv/pyenv) is the original repository and documentation.
    - Also [this](https://github.com/pyenv/pyenv-virtualenv) is the same repository install the oficial virtual environment management tool.
    - So I think I have nailed it. It is also working very well with vscode.

- Now to create a contrrol environment. I have insttalled 3.7.2 under pyenv. So we are expecting to get positive result with this config. Let's see.
    - Unfourtunatelly it is negative. I have installed the python packages from requirements. This leads me that the requirements python module versions has wrong verrsions. Since we got negative results from localdev which we also installed frrom requirements.
    - The next plan is to switch back to working environment get the requirements and compare them.
        - Ok at first glance there is bunch of differences. However, I will concantrate on tensorflow version. Because in the cloud I know tf 2.2 is neg and tf 2.1 is positive. So it is the case here also. YES! we got positive with tf 2.1
        - So I am going to freeze the environment with tf 2.1 and continue on that version. Let's copy this to localdev GPU enabled environment.
    - YES! we have positive in localdev even though the python vversion was not 3.7.2. it was 3.6.9. 

- Let's sweep it baby! [sweep](https://app.wandb.ai/hakanonal/minibar/sweeps/a7qyh03m/overview)
- I have deleted the agg sweeps with the run to properlly see the imprrovements of the aggragated data versiion.

#### 20.07.2020

- All my god. This ia big milestone for me to celebrate. I have created a network that validation acuracy is more than 0.95. And there is lot's of them. https://app.wandb.ai/hakanonal/minibar/reports/The-Result-of-The-Simple-CNN-Clasification--VmlldzoxNzQ3Mjk
- What's Next?
- Ok let's check this model with test data.
    - since we did not save any of these sweep runs. (because model is too big and I do not want to loose time to upload all of them.) I had to start a single run with best hyperparameters and capture the model.
    - Meahwhile I have  written a (test.py) script that can you can evaluate the never seen test data. I am waiting the single run to finish.
        - I am planing to add a confusion matrix
    - The problem I have encountered is: I have succesfully trained a model however when I am feeding the data via generator I do not know the class indices' label. Generator did that for me automaticlly and I can not get back the class names. So fo the simple_cnn_classification version (for every model that has fixed output dimensions.) I need to give static list of classes so that when the model has been trained I can re find the correct label of the prediction when model is in action. 

#### 21.07.2020

- So I am starting the day with fixing the class labels. 
    - Before doing that I want to see how well the evaulation is going with current messed up class indeces.
    - I want to also build a confusion matrix to be able to compare before and after. Afterall my assumption above may be dull.
    - While I was checking [this](https://github.com/keras-team/keras-preprocessing/issues/289) discussion I have come realized that flow object has class_indices property. And it seems that it is like alphabetical order. I have used this.
    - I have deleted the (test.py) and created (evaluation.ipynb) to discover the evalutaion more flexiably.
    - I have used [this](https://www.geeksforgeeks.org/matplotlib-axes-axes-bar-in-python/) article to plt bar chart on multiple dimensions on same plot.
    - I have used [this](https://www.kaggle.com/jprakashds/confusion-matrix-in-python-binary-class) article to plot confusion matrix
    - So to sum it up we do not have problem of classes that has been messed up. As long as we read classes via ImageDataGenerator the classes are going to be ordered alpabeticlly and we will be evalitate the model with different test dataset which has not been seen by the model before for training.
    - There is no need to fix the clases right now.

- Well I have managed to evaluate a sample for the model via (evaluation.ipynb), however I would like to add a confusion matrix that the total number of correct and wrong prediction for each product. So that we can see if there is a fine tune in terms of product. How can I do that?
    - Ok [this](https://www.python-course.eu/confusion_matrix.php) article very helpful in terms of understand the underhood of recall precision terms.
        - When I read that article it explains the confusion matrix in terms of guesing classes in terms of other classes.
    - I have checked if there is already lib that can accomplish task bump into [this](https://stackoverflow.com/questions/43076609/how-to-calculate-precision-and-recall-in-keras) and [this](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report)
        - So in theory if I add all the ground truth counts and prediction counts I can use the sklearn classification_report tool. Let's try
            - side note: it turns out the if you itarrate trough a keras image data generartor it will loop forever. However you can get the size via len()
            - Total catastrophy: classification_report this report outputs 0 for all classes.
            - Well it turns out that it is a report for only images that includes only one object.
            - So geetting the total numbers is totally nonsense I am deleting it.
    - Well I can make a cofusion matrix for a single sample but how can evaluate the entire test set in terms of each product prredicted.
        - I will think about [this](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html) article

- I have added evaluate_generator call for test data set only for now.

#### 22.07.2020

- Today I want to start the day with trying the following [article](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html) with data and try to underrstand the result
    - I think I have reasonable data to analyze. So you see with this function, I can 40 different confusion matrix for each class/product.
    - My initial attempt that it seems to me the numbers are very "confused" :) So I have hyptized that since I am collecting prediction and ground truths seperatelly, the there is actually different results of differrent samples. So I am going to collect ground truths and prediction within a single loop. I am assuming that it is going to be slow process.
        - It is pretty fast though and I have a very satisfying result after I have ploted them with matplotlib
        - The result is also very insightful. I have added accuracy precision and recall for each class/product
        - I think I have evaluated the model perfectlly. I have also put the recall prercision data into table and sort them.

#### 23.07.2020

- To sum up my work up-to date (1 month)
    - I have treied to create a simple image classifer.
    - Find out that it is very slow and decided to move into GPU environment.
    - Accomplish to create GPU environments on local and cloud.
    - Tried to improve metrics with different networks using pre-trained ones VGG16
    - Solved missed network parameters.
    - Learned more about installing ubuntu and machine learning environments.
    - Learned more about establishing multiple remote ssh connections.
    - Learned more about how to discover data
    - Learned about pyenv virtualenv working principles.
    - Accomplished to fully evaluate a trained image classifier.

- So since we have succesfully created simple image classifer, I would still like to see how it would be trained if we used pre-trained clasifers. So today I am planing to work on (pretrained_vgg_classification.py) I am deleting the file (deeper_vgg_classification.py) since it was a desparete branch back then to increase the metrics. Since we have solved the problem with the metrics there is no need that branch.
    - I will take the simple networks top dense layers to this new vgg16.
    - Alright I have started a new [sweep](https://app.wandb.ai/hakanonal/minibar/sweeps/f38vw6n2)
        - In the first couple of epochs it's initial metrics are performing better than simple classifier.
        - Side Note: When I compare the GPU power consumption/temp etc the new sweep has signifcantlly working harder. TThis outcome may be normal because we are trraining a much morree deeper convolution layers.

- Meanwhile when I think in my "thinking room" I have relized that I am comparing the sweeps in terms of accuracy and loss. Hopwever as seen in the evaluation notebook, recall and precision is a little bit lower than accurracy. So it is good to research on comparing the runs on using different metrics. my next move will like that.
    - I am starting this topic with a little bit research. I want to understand how keras metrics classes works? I also want to understand how accuracy metric is really working?
        - I am lost already!
        - I have learned what [one-hot](https://en.wikipedia.org/wiki/One-hot) is.
        - I am reading [this](https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class) in detail.
        - I have learned the difference between categorial cross entropy and sparse cattegorial cross entropy from [here](https://stats.stackexchange.com/questions/326065/cross-entropy-vs-sparse-cross-entropy-when-to-use-one-over-the-other)
        - Somehow I find myself looking into loss functions. What I have understand is changing the metric should not change the way network learns. Because training process tries to minimize the loss by optimizing the weights and biases.
    - Ok how about trying to give multiple metrics to the compile method and see what happens.
        - Well I am very excited now since precision and recall metrics seems to be already exists
        - Well I have included the metrics precision and recall to the compile method but tthe wierd thing that when I add the accuracy with class Accuracy() It show huge difference if I added not class but by string.
            - Yeah [this](https://neptune.ai/blog/keras-metrics) helped a lot! Seems that the key word accuracy decides best acuracy according to your network shape bla bla! How nice. So if that is true then if I use the BinaryAccuracy or other accuracy metrrics see similar results with the expresion parameter.
                - binary_accuracy is positive
                - CategoricalAccuracy is negative
                - SparseTopKCategoricalAccuracy give error problly the shape is invalid to be a sparse.
            - Well I have added bunch of metrics let's see if this is going to sync with wandb. I have to wait at least 1 epoch bout 15min on my local CPU enabled machine.
                - Oh yeah we have bunch of metrics now. Let me organize the dashboard.
                - I am going to stop the last sweep and upgrade the new metrics so that we can compare models better.


#### 26.07.2020

- Just checked the dashboard on the status of the [sweep](https://app.wandb.ai/hakanonal/minibar/sweeps/f38vw6n2).
    - I have grouped the runs in terms of learning rates. Lower rates are seems to be learn fast enough and converges to the best metrics among others. Here is the [report](https://app.wandb.ai/hakanonal/minibar/reports/Pre-trained-VGG16-Network-Grouped-by-Learning-Rate--VmlldzoxODMyNDQ) for this analyze. 

#### 27.07.2020

- I am having hard time to understand and select the base run. Suprisinglly, although most of the runs have very good precision and recall, I do not understand why mean io is under 0.5 I need to resarch on that more.
    - What if I create a new metric exploration notebook just to explore metrics class of keras. I can test it with a sattic data with single neuron network. And see what's hapening. Let's try
        - So I have created a new notebook just to explore hoow metrics are working. I have downloaded a simple dataset and construct a very simple model. The results seems to be very similar. At least I get high precision and recall but not high mean io. So by exploring arround this easy sample I can understand how the metrics are working.
    - I have decided to park this area for later.

#### 28.07.2020

- I have several next steps:
    - I want to see different pre-trained network like resnet and compare it with vgg16.
    - I want to create a new network that is not a classification network but a regresision network that predicts number of objects and/or class id of objects.

- Resnet
    - Reading [this](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035) article
    - Checked the versions of resnet from [here](https://keras.io/api/applications/)
    - Let's move with ResNet50V2. I will try toı discover the input and output on my dataset discovery notebook.
    - ResNet needs to preprocess images before feeding in to the network. [This](https://stackoverflow.com/questions/50133385/preprocessing-images-generated-using-keras-function-imagedatagenerator-to-trai) article shows parameter of preprocessing.
    - So I created a new script called (resnet_classification.py) I have configured the sweep script. Differently I have decided not to sweep different epochs. I have fixed the epoch to 200. 
    - The new script's initial metrics on my local CPU machine is very similar with others. Hopefully I have done it right! We'll see. [sweep](https://app.wandb.ai/hakanonal/minibar/sweeps/4o1a9cyv)

#### 29.07.2020

- Last [sweep](https://app.wandb.ai/hakanonal/minibar/sweeps/4o1a9cyv) has some problems. 4th [run](https://app.wandb.ai/hakanonal/minibar/runs/ufj9aubl/logs) has given an error.
    - What I have understand is that GPU memory may not be enough to train resnet.
    - [this](https://stackoverflow.com/questions/50760543/error-oom-when-allocating-tensor-with-shape/50764934) is very insightful "OOM stands for Out Of Memory. That means that your GPU has run out of space, presumably because you've allocated other tensors which are too large. You can fix this by making your model smaller or reducing your batch size. By the looks of it, you're feeding in a large image (800x1280) you may want to consider downsampling."
    - Among preior runs the crashed run has 14 batch size which is the highest among others. So I will set up a new sweep that limits the batch size up most 10 unfourtunatelly. :D
    - We are moving on...

- Ok so I have decided to check this mean_io metric again. To do that I have downloaded one of the pretrained vgg models from local dev environment. I have pluged it to evalutaion notebook.
    - First I had to solve the problem of compile. It seems that it can not load the model with tthe metrics precision. So I have give the load_model method the compile parameter false, and I compile it without precision metric. So we have done with that problem. 
    - Second I am feeding the test data. However, the pretrained vgg data's height and width seems to be swtiched places. I could not understand why it is the case. I could not find the reason but temprolaraly I have switched them in the generator.
    - Oh yeah! Hit to the wall hard. I have understood the reason why heaight and width has swtiched. Because I have configurered the yaml wrong. They are switched somehow. So all sweep trains are actually in wrong image sizes.
    - By the way I have added the mean iou metric to the evalutaion notebook and it seems very consistant. with the results seen in wandb. I am pretty sure that all metrics are working well. Rolling back I have relized that I have made mistake. When I correct it the mean_iou in evaluation notebook is way above the ones that we observe in wandb.
        - I will try to compile the recall precision and mean iou into the model evaluate with single line and see what is going to happen.
            - Yes this result that mean iou is about 0.4 which is consistant with wandb. So there is a problem calculating mean iou with keras class.
    - I have manually calculated the all confusiton matrix of each class. The overall precision and recall values are matches with the function evaluate brings. 
        - When I checked the [documentation](https://keras.io/api/metrics/classification_metrics/#truepositives-class) of mean iuo of keras, it writes the formula that I want in the descripttion, but in the example the calcultaion seems different to which I do not understand.
    - I have added the metrics classes of TP,TN,FP,FN and the result is matches with that I have calculated manually. So I need to calculate the mean iou the way I want. TP / (TP+FN+FP)
    - Here are some articles that I have researched about metrics. It seems that mean IOU is not correct metrics for me because we do not identify the objects' area but we only clasify them.
        - https://github.com/Cartucho/mAP
        - https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
        - https://medium.com/@pds.bangalore/mean-average-precision-abd77d0b9a7e
        - https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52 this is a llitle bit easiesr to understnd. It explains the classificattion and object detetction seperatelly.
        - https://towardsdatascience.com/what-is-mean-average-precision-map-in-object-detection-8f893b48afd3
    - Avarage precision seems to be the correct for us but I need to understtand it better.
    - I am trying other metrics in keras metrics api.
        - It seems that there are some missing parrameters in the implementtation. I think my installed version of keras and/or tensorflow is not matching.
    - I will create a new environment for latest keras tensorflow.
        - I have installed tf 2.2 with the environment minibar-tf2.2 
            - When I try the evaluate with this version the accuracy has gone 0.24
            - The AUC metric has now multi_label parameter. This means it does not exixts in tf2.2
        - I have installed tf 2.3 with the environment minibar-tf2.3 
            - Has similar results.
        - AUC does not meet my creteria. And moreover I did not understand how it works.

- In this new circumsttances I am planing to wipe all runs and align the metrics better. restart to sweep with simple, vgg16, resnet respecttivlelly.
- To conclude today, I will try to construct my own metric called: TP / (TP+FN+FP)
    - I have decided that I will not try to understand AP and/or mAP now. For this project precision recall and TP / (TP+FN+FP) (I am not sure what this metric is called now! I am confused) will be sufficent to measure the object/product classficiation.
    

#### 5.08.2020

- My Own TP / (TP+FN+FP) Go...
    - reading [this](https://keras.io/api/metrics/#creating-custom-metrics) article. It is about creating custom metrics in keras.
    - So I am trying to mimic how precision is coded and modify according to my needs. [Source](https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/metrics.py)
    - So with minor changes of [this](https://github.com/keras-team/keras/blob/2.3.1/keras/metrics.py) code on the class of Precision I have created a script called (helpers/overallperformance.py) which I know it is teriable name. But at least I will be able to measure the combined performance of recall and precision.
    - I have tested in evaluation notebook that confirmed the correct number with calculation manually.

- So I want to give a way new runs with GPU with this new metric completelly compare the runs. I will delete all previous runs.
    - I can not access to my local GPU right now. until then I will start this on cloud GPU.
        - The cloud GPU has conda environment so I will install the same environment with my local GPU to run in parallel in later.
        - like pyenv [this](https://towardsdatascience.com/environment-management-with-conda-python-2-3-b9961a8a5097) article is very helpful to do basic virtual environment stuff in anaconda environment.
        - So I have created an environment called minibar on my cloud server using conda environment.
        - I have also changed the cuda version using [this](https://docs.aws.amazon.com/dlami/latest/devguide/dlami-dg.pdf) article on section "configuring cuda versions"
        ```
        $ sudo rm /usr/local/cuda
        sudo ln -s /usr/local/cuda-10.1 /usr/local/cuda
        ```
    - the [sweep](https://app.wandb.ai/hakanonal/minibar/sweeps/6an909zv) has started

#### 06.08.2020

- Oyy! I have forgotten to plug the computers' electric power so that the battery has finished. That's why I could not access to my local GPU
    - I am going to start the same sweep also inmy local GPU to run in parallel and faster.

#### 11.08.2020

- Well it's been 5 days for the simple classification. 109 runs has been completed. Pretty good. 
    - It is certeainly clear that lr 1e-4 is the winner. 
    - However, when I check the batch sizes I can not see a consistant result there are both good and bad results in varius batch sizes. And when I say bad I mean total catastrpohy (zero)
    - Moreover, the best TP / (TP+FN+FP) is not above 0.83
    - I see that the sweep has not tried all posiible batch sizes yet. I would be morre confident if it is going to work other batch sizes.

- Last night I had a talk with my friend. about genetic algorithms. Curentlly I am brute forcing all posible hyper parameters, via sweep and all runs are initially random. So all trains are being trained from the begening. So I have an intension to apply the genetic algorithms to train the follow up training. This means that I need to apply to continue training inherting from previous generations. I will research about that.
    - Additional on this idea! curentlly I am discovering only different batch sizes and learning rates only. Also I am trying to compare 3 different networks. And that's it. However, **wouldn't be nice if I could have create a single script that explores all different kind of networks incdluding much more varius hyper parameters additional on to the batch size and the learning rate.**

#### 24.08.2020

- Today I have completed the all runs and created a report [here](https://app.wandb.ai/hakanonal/minibar/reports/Comparison-of-SimpleCNN-VGG16-ResNet--VmlldzoyMTQ2NjE?accessToken=ne38hb69rclmjn2zgen0hsiy8t9t8adlgb6a1o0a4lxhven3mgj3njlijo57b9jb)

- I have several next steps: Moved these steps to a project board [here](https://github.com/hakanonal/minibar/projects/1)
    - Done! I want to see different pre-trained network like resnet and compare it with vgg16. report is above.
    - Show the corolation between the number of classes appeared in samples and the overall performance.
    - Design a system that feeds new data to enrich the training dataset.
    - Create a live environment that can be accessed by the other project components.
    - I want to create a new network that is not a classification network but a regresision network that predicts number of objects and/or class id of objects.

- Show the corolation between the number of classes appeared in samples and the overall performance. [card](https://github.com/hakanonal/minibar/projects/1#card-44179839)
    - I have began it by downloading the latetest resnet run which has performed over 0.83
    - Oh yeah when I first evaluate the resnet network the performance was terriable. However I have just remembered that the samples has to pre processed for resne and my evaluation script does not do that. Let first do it.
        - Yeah that has fixed!
    - So how can we show the corrolation between the number of appreances of class/product in samples. 
        - So in (evaluation.ipynb) notebook I have tried to plot the report dataframe on the same plot. To see if there is a corrolation between the appereance of the count of the class/product and the overall performance. The result is in [this](evaluation.ipynb) notebook. Unforuntunatelly, there is no direct correspondance. 
            - It is obvious that some classes has very little samples and they have to be included more samples
            - but the classes/products that has over 78 (that is the minimum count (damla su) that has decent amount of samples.) gives the overall performance more than %70. 
            - There is clases/products that has less then 500 appereances gives high performance
            - On the other hand best overall performance of class efes_malt has less then 400 appereance and it has significantlly high performance then others.
    - To conclude: there is definatelly other parameters then class/product appreance count on traning set that effects  the overall performance.

#### 26.08.2020

- Today I am spending some time to understand better how the original minibar-master resnet object detctor project has been contructed.

- [Regression](https://github.com/hakanonal/minibar/projects/1#card-44179863) I am starting to create a new network that is not a classification network but a regresision network that predicts number of objects and/or class id of objects.
    - Research
        - Where to start! base rememberaing about [regressions](https://en.wikipedia.org/wiki/Regression_analysis)
        - Someone like me wants to convert image classifer into object counter [here](https://stackoverflow.com/questions/59472360/classification-vs-regression-for-object-counting-with-convolutional-neural-netw)
        - [This](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5426829/) seems very helpful. It counts the number of totamtos(red dots) on the image. There bunch of details about the network detail.
    - So my plan is to auto convert the dataset that counts the number of objects in image and train with that. 
        - We are already doing that in out utility function called decouple
        - I have created a util function that converts the matrix into dataframe consists of filename and count column names
        - I am modifying the Imagedatagenerator class mode to sparse by reading the reference [document](https://keras.io/api/preprocessing/image/)
        - changed the last layer to 1 nouron only with no activation function.
        - Since it is a regrression problem I have changed the loss function to mean squered error.
        - and metrics should be very simple right? we need only accuracy and loss is enough.
        - class_mode sparse did not worked I got error class_mode="sparse", y_col="count" column values must be strings.
            - [this](https://stackoverflow.com/questions/41749398/using-keras-imagedatagenerator-in-a-regression-model) article tells me that class_mode other is for regression models in version 2.2.4. I am not tusing that version. but I will try.
                - nope! error: Invalid class_mode: other; expected one of: {'multi_output', 'sparse', 'categorical', 'input', 'raw', None, 'binary'}
                - same article has accepted answer frlow from directory. I will use sparse but I will construct the dataframe count column cast to string and let's see I got the error.
        - Yeap I got rid of the error. however I am not sure the data generator is working good. I will discover it better.
            - Well when I discover the data I see that the generator again re-classes the output. For example count 1 images outputs as 0 as if they are the 0th class. However we want the real value right away.
            - [This](https://www.kaggle.com/lgmoneda/data-augmentation-regression) example notebook tried to count the number coins in the picture. It has used the regular flow method.
            - When I read the reference document again I have solved the problem. class mode should be raw. Since it is int I should roll back to add int value (not casting to string)
        - Yes now I am confident to deploy it to my local GPU.
            - I have started a single [run](https://app.wandb.ai/hakanonal/minibar/runs/71kv8jif) on GPU. However wandb dashboard is not suiatable for this kind of regression problem maybe I can create a new workspace for that.
                - No new workspace but only speration of standart metrics and confusion matrix metrics did the job.

#### 27.08.2020

- Last [run](https://app.wandb.ai/hakanonal/minibar/runs/71kv8jif) has finished in ~6hrs. How can I evaluate?
    - I have created a new notebook just copied the evaluation as eval_regrerssion
        - [0.3180391490459442, 0.5643564462661743] which one is accuracy and which one is loss. either one loss seems under 1. very impresive!
        - Well I have tried to evaluate the picturres one by one as I did normal clasification evaluation. Genereally it finds the right number. Event It is not accurate it is one off.
        - It is weird that [0.03913490101695061, 0.5643564462661743] this evaluate function gives different results for different times. Which is not expected.
        - It is also very weird that the training performance way much worse than these metrics. I should again manually calcaulate the metric and check if it is giving the same result oor not.
        - So I am thinking to evaluate this regression in terms how accurate each sample is find. 
            - I can measure it by find the right number or not with confusion matrix and all. 
            - And/Or if it is not same with ground truth how off it is. I bealive mean squered error should already give that to me. if it is working corectlly I do not think to find the confusion matrix. Since if loss ise below 0.5 then it means it will generally rounded to the correct ground truth. Hoowever, I am not sure yet...

#### 14.09.2020

- And after a very long break on this project. I have come back! Hello again...
- I have spent some time to read my journal to remember.
- I have checked https://chooch.ai 
    - I had signed up this service before. today I had chance to explore a little bit more. 
    - There are bunch of pre-trained models in the app.
    - It is possible to send images or videos and clasify them with the pretrained models. It is workiing impresivelly fast.
    - I have cheked image clasification and object detection panels from the dashboard. Since we have multiple objects in a single image, the image clasification panel seems to be does not allow that. Hence it creates classes as folder and wants you to upload images under that classes. In this case I need to upload same picture into different classes. To test it fully I need to fully redistribute all pictures into class folders. So that I can upload them folder/class by folder/class. I have parked this to aside for now.
- I want to continue on regression problem!
    - I am remembering my last notes.
    - I have go through (eval_regresseion.ipynb) notebook.
    - Is keras evalutate results consistant?
        - First run [0.03128417208790779, 0.5643564462661743]
        - Second run [0.011685364879667759, 0.5643564462661743]
        - second number seems to be consistant. It is consisttant also with the runs that I have conducted 15days ago.
        - I have read [this](https://stackoverflow.com/questions/51299836/what-values-are-returned-from-model-evaluate-in-keras/51303340) model has metrics_names method. When I run the method I have understand that first number is loss and second number is accuracy. I do not understand why the loss is different. Sinc accuracy is always same.
        - Severral runs to log
            - [0.14309091866016388, 0.5643564462661743]
            - [0.6417807340621948, 0.5643564462661743]
            - [0.10828681290149689, 0.5643564462661743]
        - Well this means we are getting the right amount of product from the model in %56 times. And the ones that we can not guess right we are only 1 off since the loss is below 1.
        - Since in my eval_regression notebook when I examine the picture by picture, I do not get conflicting results with the values that have found with evalute metthod. I will nott dig down deeper on this. For now.
    - So I am wondering two things.
        - Should I modify the model to find the number of objects for each product/class?
            - If yes I wonder if the data has multiple objects for the same product/class in the same picture?
                - I wonder this becuase if every product appears only once in a sinlge picture then the dataset would have no different then the one we have used for classification problem. Every product/class is going to be has 1 in one hot encoded output. So I am planing to discover this situtation in (dataset_discovery.ipynb)
        - Should I improve the accuracy. If yes how could I improve accuracy? which is a very big question.

#### 15.09.2020

- Today I want to start to discover data and find that if there is more than one appearance of a single product/class in a single picture.
    - I have tried to discover if a product/class appears more than once in a single image. I have found that we have only handful of those samples. they are listed in (dataset_discovery.ipynb) notebook. 
    - When checked all of them  manually one by one almost all of them are actually mis-labeled.
    - So to conclude there is a big evidance that the data has some mis-labeled data which may acutally decreases the accuracy and performance.
        - How can we somehow detect mis-labeled data?
- I have decided to prepeare the resnet_regression model for each class ready and put aside the regression [card](https://github.com/hakanonal/minibar/projects/1#card-44179863). After that I will ready to get back in touch with the project team and find a common way to increase the training data and also corrrect the mis-labeled data.
    - So to do that I need to again understand the process of keras ImageGeneration.
    - [This](https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24) article helped me to understand the multi labled classes. So I will convert matrix to dataframe for each class seperate column. And the class_mode iis again going to be raw because we are solving a regression problem.
        - I have copied the resnet_regression.py script with the classed version. And also created a verrsion of matrix_to_df that outputs each seperate product/class as seperate columns. It is ready to deploy to local GPU
            - It has started [run](https://app.wandb.ai/hakanonal/minibar/runs/49td0rbw)
            - Meanwhile we can work on evaluation notebook.