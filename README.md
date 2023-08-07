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
- [License](#license)
- [Acknowledgments](#acknowledgments)

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
- nltk

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

- Embedding layer
- Dropout layer
- Bidirectional LSTM layer
- Bidirectional LSTM layer with recurrent dropout
- TimeDistributed Dense layer with softmax activation

