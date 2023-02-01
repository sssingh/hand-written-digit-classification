# MNIST Classification using Neural Network 
In this project we deal with a `multi-label-classification` problem where we classify hand-written digits images in MNIST dataset using a custom built neural-network

<img src="https://github.com/sssingh/hand-written-digit-classification/blob/master/assets/mnist_intro.jpeg?raw=true">

## Features
⚡Multi Label Image Classification
⚡Cutsom Fully Connected NN
⚡MNIST
⚡PyTorch

## Table of Contents
- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [Evaluation Criteria](#evaluation-criteria)
- [Solution Approach](#solution-approach)
- [How To Use](#how-to-use)
- [License](#license)
- [Get in touch](#get-in-touch)
- [Credits](#credits)

## Introduction
The Modified National Institute of Standards and Technology (MIST) dataset was prepared by combining two National Institute of Standards and Technology (NIST) databases. These databases contained the images of hand-written digits by high school students and the United States Census Bureau employees. The MNIST dataset is widely used in academia for training and testing machine-learning models. I tend to think of this as the "Hello World" for classification tasks to experiment with various ML and DL models. The MNIST digit classification is possibly one of an aspiring ML/DL learner's first tasks.MNIST is an excellent dataset for people who wish to try machine learning techniques on a real-world dataset with very little time and effort on data preparation and pre-processing, which is the most time-consuming task in data science projects.   


## Objective
We'll build a neural network using PyTorch to discriminate between digits 0 to 9 in the MNIST dataset.

## Dataset
- Dataset consists of 60,000 training images and 10,000 training images.
- Images may belong to any of the ten classes (digits 0 to 9)
- Each image in the dataset is a 28x28 pixel grayscale image, a zoomed-in single image shown below...
- Dataset consists on 60,000 training images and 10,000 testing images.
- Images may belong to any of the 10 classes (digits 0 to 9)
- Each image in the dataset is 28x28 pixel gray scale image, a zoomed in single images shown below...

<img src="https://github.com/sssingh/hand-written-digit-classification/blob/master/assets/mnist_single_image.png">

- Here are a few more samples of other digits images from the training dataset with their respective labels...

<img src="https://github.com/sssingh/hand-written-digit-classification/blob/master/assets/mnist_samples.png">


- We will use the in-built MNIST dataset from PyTorch's `torchvision` package. The advantage of using the dataset this way is that we get a clean pre-processed dataset that pairs the image and respective label nicely, making our life easier when we iterate through the image samples while training and testing the model. Alternatively, the raw dataset can be downloaded from the original source [here](http://yann.lecun.com/exdb/mnist/). The raw dataset comes as a set of zip files containing training images, training images, testing images, and testing images in separate files.

## Evaluation Criteria

### Loss Function  
Negative Log-Likelihood Loss (NLLLoss) is used as the loss function during model training and validation 

<img src="https://github.com/sssingh/hand-written-digit-classification/blob/master/assets/logsoftmax.png?raw=true">

<img src="https://github.com/sssingh/hand-written-digit-classification/blob/master/assets/nllloss.png?raw=true">

<br>Note the `negative` sign in front `NLLLoss` formula (and in the `BCELoss` formula as well) hence negative in the name. The negative sign is put in front to make the average loss positive. Suppose we don't do this then since the `log` of a number less than 1 is negative. In that case, we will have a negative overall average loss. To reduce the loss, we need to `maximize` the loss function instead of `minimizing,` which is a much easier task mathematically than `maximizing.`

### Performance Metric

`accuracy` is used as the model's performance metric on the test-set 

<img src="https://github.com/sssingh/hand-written-digit-classification/blob/master/assets/accuracy.png?raw=true">

## Solution Approach
- Training dataset of 60,000 images and labels along with testing dataset of 10,000 images and labels are downloaded from torchvision.
- The training dataset is then split into 20% of the validation set (12,000 images) and 80% of the training set (48,000 images)
- The training, validation, and testing datasets are then wrapped in PyTorch `DataLoaders` objects so that we can iterate through them with ease. Again, a typical `batch_size` 32 is used.
- The neural network is implemented as a subclass of the `nn.Module` PyTorch class. The network has an input, two hidden layers (784 and 128 nodes), and an output layer with ten nodes. Hidden layers use `ReLU` activation function. Output layer uses `LogSoftmax` activation. A 25% dropout regularization is used after each of the hidden-layer outputs. Images of size 28x28 are flattened before being fed to the network as a 784 element long vector. A high-level network schematic is shown below...

<img src="https://github.com/sssingh/hand-written-digit-classification/blob/master/assets/network.png?raw=true">

- Network is trained and validated for ten epochs using the `NLLLoss` function and `Adam` optimizer with a learning rate of 0.001.
- We keep track of training and validation losses. When plotted, we observe that the model starts to `overfit` around the 5th epoch.

<img src="https://github.com/sssingh/hand-written-digit-classification/blob/master/assets/loss_plot.png?raw=true">

- During the validation, we compare the current validation loss with previous validation loss and save the model of validation loss has `decreased` further; this way, we'll end up with the best model kept and not the model from the last epoch, which could be a model that overfits (as we can see from plot above)
- The trained model is then evaluated on an unseen test dataset. For this, we first load the saved model and then predict over 10,000 testing images.
- For each digit label, we keep track of prediction accuracy as correct-prediction/total-number-of-images. As a result, the network can achieve around `97.52%` accuracy. The test result summary is shown below...

<img src="https://github.com/sssingh/hand-written-digit-classification/blob/master/assets/test_results.png?raw=true">

## How To Use
1. Ensure the below-listed packages are installed
    - `NumPy`
    - `matplotlib`
    - `torch`
    - `torchvision`
2. Download `mnist_classification_nn_pytorch.ipynb` jupyter notebook from this repo
3. Execute the notebook from start to finish in one go. If a GPU is available (recommended), it'll use it automatically; otherwise, it'll fall back to the CPU. 
4. A machine with `NVIDIA Quadro P5000` GPU with 16GB memory takes approximately 5-7 minutes to train and validate for ten epochs.
5. A trained model has been provided as part of this repo, `MNIST_model.pth`. The trained model can be loaded and used for prediction, as shown in the below code snippet... 

```python
    # Load model with the trained wights
    weights = torch.load('MNIST_model.pth')
    model = Network()
    model.load_state_dict(weights)
    # move model to the available device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    proba = torch.exp(model(image)) 
    _, pred_label = torch.max(proba, dim=1)
    print(pred_label)
```

## License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Get in touch
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/sssingh)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/_sssingh)
[![website](https://img.shields.io/badge/website-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://datamatrix-ml.com/)

## Credits
- Title photo by [Ales Nesetril On Unsplash](https://unsplash.com/photos/ex_p4AaBxbs?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink)
- Data collected and analyzed by [Worldline and the Machine Learning Group](http://mlg.ulb.ac.be) 
- Dataset sourced from [Kaggle](https://www.kaggle.com/)

[Back To The Top](#MNIST-Classification-using-Neural-Network)
