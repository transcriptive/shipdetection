#  Automated Ship Detection using Satellite Photography

## Problem Statement
Using computer vision algorithms, develop a model to automatically identify the presence of ships in satellite images.
 
## Table of Contents
1. [eda.ipynb](https://github.com/transcriptive/shipdetection/blob/main/eda.ipynb) — Exploratory data analysis.
2. [data-preprocessing.ipynb](https://github.com/transcriptive/shipdetection/blob/main/data-preprocessing.ipynb) — Initial creation of training set.
3. [mask-maker.ipynb](https://github.com/transcriptive/shipdetection/blob/main/mask-maker.ipynb) — Function to generate mask images from run-length-encoded data.
4. [u-net.ipynb](https://github.com/transcriptive/shipdetection/blob/main/u-net.ipynb) — training for u-net used to generate masks.
5. [ship-or-no-ship](https://github.com/transcriptive/shipdetection/tree/main/ship-or-no-ship) — subdirectory containing the dataset generator and CNN to detect presence of ships in the imagery.

## Executive Summary
Airbus offers a service to monitor maritime traffic using aerial and satellite imagery. This is useful for agencies which oversee ports, as well as government agencies responsible for territorial waters. Aerial monitoring can identify unregistered/illicit maritime traffic such as piracy, illegal fishing, drug trafficking, and illegal cargo movement. In 2018, they launched a Kaggle competition to find solutions to automate this process. This repository contains two neural networks designed to address this issue. A Convolutional Neural Network is used to detect the presence of ships in a given image, and a U-Net network is used to generate bounding boxes for the images which contain ships. Due to computing limitations, the U-Net was trained on a subset of 1000 images, and the CNN was trained on a subsect of 2000 images.

## Data Sources
This project used data available at [this Kaggle page](https://www.kaggle.com/c/airbus-ship-detection/data).

## Conclusion and Recommendations
While I was able to achieve a result that is higher than the baseline in terms of predicting whether a given image has a ship present, there is still room for improving the prediction accuracy. In order to improve performance, I would suggest the following steps:
- Using a larger training set on a more powerful computer. Due to the size of the iamges (768 * 768) and the fact that several images contained ships so small that resizing the image would obfuscate their presence, the amount of memory required to train on these images is very large.
- Manually segment a subset of the images based on the size of the ship and use this to train a network to detect the size of the ships in an image. Then, train a series of networks to achieve optimal performance for small ships, medium ships, and large ships. This would address the issue where the network was able to correctly identify large ships, but struggled when identifying smaller ships.
- Explore options to automatically adjust the contrast of an image to help accommodate for situations where cloud cover or other visual conditions make the ships hard to identify.
