# Image Captioning Project

## Overview
This project implements an Image-Captioning model that generates captions for input images. The implementation leverages PyTorch for building and training neural network models.

## Installation
To set up the project environment:
```bash
pip install -r requirements.txt
```
or run setup.sh

## Usage
To run the project:

Modify the configuration file to specify paths and settings for datasets and model parameters.
Run the train.py script to start the training process:
```bash
python train.py --config configs/configFC.yaml
```
or
```bash
python train.py --config configs/configAttn.yaml
```
```bash
python train.py --config configs/configTransformer.yaml
```

## Weights

Download weights from https://cityuni-my.sharepoint.com/:f:/g/personal/youssef_shaarawy_city_ac_uk/El_55fvjKWdMg6YdT3EhtUwBMMcceq5atQYuZQLpIyfrbw?e=xbYE7E

Add them to the weights folder.

## Dataset

Download dataset from kaggle: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset