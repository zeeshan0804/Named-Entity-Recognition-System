# Named Entity Recognition (NER) System

This project implements a Named Entity Recognition (NER) model using TensorFlow. The model is trained to identify named entities in text data.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [About Flask Web App](#about-flask-web-app)

## Introduction

This project demonstrates the implementation of a NER model using TensorFlow. The model is trained to recognize named entities (such as names of people, organizations, locations, etc.) in text data.

## Getting Started

### Prerequisites

Make sure you have the following packages installed:

- Python
- TensorFlow
- pandas
- numpy
- scikit-learn
- matplotlib
- Flask

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/zeeshan0804/Named-Entity-Recognition-System.git
   cd Named-Entity-Recognition-System
2. Install the required packages:

   ```bash
   pip install -r requirements.txt

## Data

The dataset used for training and evaluation is sourced from `Dataset/ner_dataset.csv` and `Dataset/ner.csv`. The dataset contains labeled named entities and their corresponding tags.

## Model Architecture

The NER model is built using a sequential architecture with the following layers:

![ner h5](https://github.com/zeeshan0804/Named-Entity-Recognition-System/assets/67948488/c91b6bf4-4459-48a4-801e-7f0d05f948e7)


- Embedding layer
- Dropout layer
- Bidirectional LSTM layer
- Bidirectional LSTM layer with recurrent dropout
- TimeDistributed Dense layer with softmax activation


## Training

The model is trained using the `train_dataset` and validated using the `val_dataset`. Training details such as loss and accuracy are logged during the training process.
After training the NER model, it's important to evaluate its performance on unseen data to assess its effectiveness. The evaluation is conducted using the validation dataset (`val_dataset`). Below are the results of the model evaluation:

- **Train Loss**: 0.0592
- **Train Accuracy**: 97.87%
- **Validation Loss**: 0.0687
- **Validation Accuracy**: 98.01%


## Evaluation

To assess the NER model's performance on unseen data, it was evaluated using the test dataset (`test_dataset`). The evaluation results on the test dataset are as follows:

- **Test Loss**: 0.06727
- **Test Accuracy**: 97.92%

## Usage

1. Install the required packages using `pip install -r requirements.txt`.

2. Train the NER model using the `NER.ipynb` script or load a pre-trained model using a `.h5` file.

3. Start the Flask web app by running `app.py`. Access the app in your web browser at `http://127.0.0.1:5000/`.

4. Enter the input text in the provided textarea and click "Predict". The predicted named entity tags will be displayed on the result page.

### Loading a Pre-trained Model

If you have a pre-trained NER model saved as a `.h5` file, you can load it using the following steps:

1. Ensure you have the model file (`ner.h5`) and the tokenizer file (`tokenizer.pkl`) in the same directory.


## About Flask Web App

The Flask web app consists of two main HTML templates:

- `home.html`: This page provides a user-friendly interface for entering text for prediction. The background color is a gradient from light blue to white.

- `results.html`: After submitting the input text, users are directed to this page, which displays the original text and the predicted named entity tags. The background color is a gradient from dark blue to light blue.

The Flask web app files are located in the `templates` directory.



