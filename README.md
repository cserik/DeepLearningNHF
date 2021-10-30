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
This dataset contains 30.000 images that we will use for training.

### Training environment
For training our model we will use Azure Machine Learning.

## Team name
veGANs

## Team members
Váli Valter\
Nandrean David Cristian\
Csató Erik
