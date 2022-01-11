# DeepLearningNHF

This repository was created by students of the Budapest University of Technology and Economics to fulfill the homework requirements of the Deep Learning subject.

## About GANs
Generative adversial networks, known as GANs are used to generate high-quality syntethic data (like images). GANs are trained in a supervised environment with two sub-models: the generator that tries to generate fake data and the discriminator that tries to classify the data as fake or real. These two are trained together, until the discriminator is fooled enough times. \
Progressive growing GANs use and updated version of the classic GAN training method that generates small images and they incrementally increase the size until they reach the target output size.
<p align="center">
<img src="/images/pg_gan.png" width="400">
</p>

## Our project
### Our goal
We will try to generate **256 x 256** synthetic images of people using progressive growing GANs.

### Dataset
We use the following dataset: https://www.kaggle.com/badasstechie/celebahq-resized-256x256 \
This dataset contains 30.000 images that we will use for training.\
The way we preprocess data has changed between milestones. 

### Training environment
For training our model we use Azure Machine Learning.

#### Milestone 1
Notebook: *DataPreprocessing.ipynb*

#### Milestone 2
Notebook: *DataPreprocessing2.ipynb*\
It uses MTCNN to extract faces from the images. This improves the training process because the network has to learn less features.
We took [this article](https://machinelearningmastery.com/how-to-implement-progressive-growing-gan-models-in-keras/) as a basis for the architecture.

#### Final
We trained our model to the size of **256 x 256** on Azure and we achieved good results. Some sample generated images and the plot of the loss function during training can be seen in the *images* folder, the synthetic image generating models can be downloaded from the *models* folder. The network architecture can be viewed in the *network_plot* folder.\
In the *azure* folder under the *scripts* folder can be seen the scripts we ran on Azure. The *crop.py* script was used to load the images and crop the faces out of them and save all in a compressed file, meanwhile the *train.py* script was used for the training process which resulted the pictures seen in the *images* folder. The *train.ipynb* notebook was used to create the train script and to start the training on Azure. You can start your own training by creating a resource group and workspace on Azure. The *config.yml* file describes the necessary environment for the *train.py* script to run on Azure.

### How to train
For training specify the number of growth phase with the *n_blocks* variable (e.g. set to 3 for 16 x 16 images).
For each growth specify the batch and epoch numbers with the *n_batch* and *n_epoch* variables.

#### Colab
Notebook: *Architecture.ipnyb*
The kaggle.json (API token for kaggle) has to be copied into *Architecture.ipynb*.
Do not load too many images (you should be fine with around 2000 loaded images) because *Colab* does not have enough memory to handle so many data.\
By executing every command in order the training process will start.

#### Azure
Notebook: *azure/train.ipnyb*
Create your own resource group and workspace on Azure upload all the images into your Datastore. Run the *crop.py* script to crop the faces out of the images and save them to a *.npz* file. Then upload the *.npz* file as well to your Datastore.\
Fill the required data in the  *azure/train.ipnyb* notebook. After that you can start your own training process on Azure!

### How to evaluate
During training generators along with some generated photos are saved each growth phase. These can be used for evaluation.\
Also you can load the models and plot your own images using the *test_model_script.ipynb*.

## Results
You can evaluate our results by looking the synthetized images
<p align="center">
<img src="/images/plot_128x128-tuned.png" width="400">
<img src="/images/plot_256x256-tuned.png" width="400">
</p>

## Android application
We created an Android application to demonstrate the capabilities of our generator model, where you can set the desired resolution and then generate realistic images of non-existent people.\
Download and install the application from *android/facegenerator.apk* to your Android device!

<p align="center">
<img src="/android/screenshots/32x32.png" width="400">
<img src="/android/screenshots/256x256.png" width="400">
</p>

Icon made by [Freepik](https://www.freepik.com) from [www.flaticon.com](https://www.flaticon.com/)

## Team name
veGANs

## Team members
Váli Valter\
Nandrean David Cristian\
Csató Erik

## Youtube video
https://www.youtube.com/watch?v=Snfwdh1CHc8
