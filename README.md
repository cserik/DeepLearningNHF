# DeepLearningNHF

This repository was created by students of the Budapest University of Technology and Economics to fulfill the homework requirements of the Deep Learning subject.

## About GANs
Generative adversial networks, known as GANs are used to generate high-quality syntethic data (like images). GANs are trained in a supervised environment with two sub-models: the generator that tries to generate fake data and the discriminator that tries to classify the data as fake or real. These two are trained together, until the discriminator is fooled enough times. \
Progressive growing GANs use and updated version of the classic GAN training method that generates small images and they incrementally increase the size until they reach the target output size.

## Our project
### Our goal
We will try to generate synthetic images of people using progressive growing GANs.

### Dataset
We use the following dataset: https://www.kaggle.com/badasstechie/celebahq-resized-256x256 \
This dataset contains 30.000 images that we will use for training.\
The way we preprocess data has changed between milestones. 
#### Milestone 1
Notebook: *DataPreprocessing.ipynb*\

#### Milestone 2
Notebook: *DataPreprocessing2.ipynb*\
It uses MTCNN to extract faces from the images. This improves the training process because the network has to learn less features.
We took [this article](https://machinelearningmastery.com/how-to-implement-progressive-growing-gan-models-in-keras/) as a basis for the architecture.

### Training environment
For training our model we use Azure Machine Learning.

### How to train
Notebook: *DataPreprocessing2.ipynb*, *Architecture.ipnyb*\
The kaggle.json (API token for kaggle) has to be copied into *DataPreprocessing2.ipynb*\.
For training specify the number of growth phase with the *n_blocks* variable (e.g. set to 3 for 16 x 16 images). **DO NOT GO ABOVE 64 x 64!**\
For each growth specify the batch and epoch numbers with the *n_batch* and *n_epoch* variables.

### How to evaluate
During training generators along with some generated photos are saved each growth phase. These can be used for evaluation.

## Team name
veGANs

## Team members
Váli Valter\
Nandrean David Cristian\
Csató Erik
