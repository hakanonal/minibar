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
    -Well It has speeded up significantlly however I got the memory error again. However it continue to train. The results seems to be same after one epoch. And no change as seen [here](https://app.wandb.ai/hakanonal/minibar/runs/2uhoeblj)

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
    - The oveerall pererformance is still verry bad. Evenworse the validattion loss seems to be incresing aftter some epochs insteead of decreasing.
    - And also it odd that 1e5 has better loss but worse acuracy. What does it mean?

- I have dicded to set-up an [sweep](https://docs.wandb.com/sweeps) in wandb

#### 14.07.2020

- sweep! To do that I need my script to convert function and make sure to call sweep and give as callback parameter.
    - the sweep has started https://app.wandb.ai/hakanonal/minibar/sweeps/slche9g5

- Meanwhile finally I have managed to setup remote connection to my local GPU development environment. Some helpful articles are below:
    - [This](https://www.ssh.com/ssh/tunneling/example) article is for ssh tunnuling options I have used remote port forwarding for my local development environment.
    - [This](https://www.ssh.com/ssh/sshd_config/) article is for sshd config. I have configured GatewayPorts config to yes in jump AWS server.
    - [This](https://medium.com/maverislabs/proxyjump-the-ssh-option-you-probably-never-heard-of-2d7e41d43464) article is for how to config the ~/.ssh/config file so that I can easiy make connection via vscode.
    - [This](https://www.everythingcli.org/ssh-tunnelling-for-fun-and-profit-autossh/) article gives info about autossh. It is needed for local development computer to keep alive the ssh connection to intermediate jump server.


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
    - In [this](https://godatadriven.com/blog/keras-multi-label-classification-with-imagedatagenerator/) article it says that if a summarize the dataframe and set the labels as array, flow_from_dataframe will handle it autmaticlly for me. Let's see is it going to work?
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
    - Also [this](https://github.com/pyenv/pyenv-virtualenv) is the same repositury install the oficial virtual environment management tool.
    - So Ithink I have nailed it. It is also working very well with vscode.

- Now to create a contrrol environment. I have insttalled 3.7.2 under pyenv. So we are expecting to get positive result with this config. Let's see.
    - Unfourtunatelly it is negative. I have installed the python packages from requirements. This leads me that the requirements python module versions has worng verrsions. Since we got negative results from localdev which we also installed frrom requirements.
    - The next plan is to switch back to working environment get the requirements and compare them.
        - Ok at first glance there is bunch of differences. However, I will concantrate on ttensorflow version. Because in the cloud I know tf 2.2 is neg and tf 2.1 is positive. So it is the case here also. YES! we got positive with tf 2.1
        - So I am going to freeze the environment with tf 2.1 and continue on that version. Let's copy this to localdev GPU enabled environment.