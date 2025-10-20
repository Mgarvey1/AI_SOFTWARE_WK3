# AI Software Engineering - Week 3 Project

A comprehensive machine learning portfolio demonstrating classical ML, deep learning, and natural language processing capabilities.

## Project Overview

This project implements three core machine learning applications:
- **Classical ML**: Iris flower classification using Decision Trees with Scikit-learn
- **Deep Learning**: Handwritten digit recognition with CNN on MNIST dataset using TensorFlow/Keras  
- **Natural Language Processing**: Sentiment analysis and Named Entity Recognition on Amazon reviews using spaCy

## Project Structure
AI_SOFTWARE_WK3/
├── task1_iris_classification/ # Classical ML with Scikit-learn
├── task2_mnist_cnn/ # Deep Learning with TensorFlow/Keras
├── task3_nlp_amazon_reviews/ # NLP with spaCy
├── requirements.txt # Project dependencies
└── README.md # Project documentation

## Features

### Task 1: Iris Dataset Classification
- Data exploration and visualization of Iris dataset
- Decision Tree classifier implementation
- Model evaluation with accuracy, precision, and recall metrics
- Feature importance analysis and dataset statistics

### Task 2: MNIST Handwritten Digit Recognition
- Convolutional Neural Network architecture design
- Image preprocessing and normalization
- Training progress monitoring with accuracy/loss plots
- High-accuracy handwritten digit recognition
- Sample predictions visualization

### Task 3: Amazon Reviews NLP Analysis
- Named Entity Recognition using spaCy's pre-trained models
- Rule-based sentiment analysis implementation
- Text preprocessing and entity extraction
- Brand and product identification from customer reviews
- Sentiment classification with entity correlation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

# Clone the repository
git clone https://github.com/Mgarvey1/AI_SOFTWARE_WK3.git

# Navigate to project directory
cd AI_SOFTWARE_WK3

# Install dependencies
pip install -r requirements.txt

Required Dependencies
scikit-learn>=1.0.0
tensorflow>=2.8.0
spacy>=3.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0

###Usage

Task 1: Iris Classification

python task1_iris_classification/iris_classifier.py

Task 2: MNIST CNN

python task2_mnist_cnn/mnist_cnn.py

Task 3: Amazon Reviews NLP

python task3_nlp_amazon_reviews/nlp_analysis.py


###Technical Implementation

##Framework Comparison

Scikit-learn: Classical ML algorithms with high-level APIs

TensorFlow/Keras: Deep learning optimized for neural networks

spaCy: NLP with pre-trained models for entity recognition


##Model Architectures

Decision Tree classifier for Iris species classification

Convolutional Neural Network for digit recognition

Rule-based system with spaCy's entity recognition


###Results


##Task 1: Iris Classification

High accuracy in species classification

Comprehensive precision and recall metrics

Feature importance analysis


##Task 2: MNIST CNN

High test accuracy on handwritten digits

Effective training with minimal overfitting

Successful prediction visualizations


##Task 3: Amazon Reviews NLP

Reliable entity extraction (brands, products, dates)

Effective sentiment classification

Context-aware linguistic processing


###Ethical Considerations

##Identified Biases

Iris Dataset: Potential overfitting due to small size

MNIST CNN: Bias toward common digit writing styles

Amazon Reviews: Limited coverage of obscure brands


##Mitigation Strategies

Cross-validation techniques

Expanded entity patterns and lexicons

Manual validation of edge cases


Contributors

Marcus-Garvey Kinyenye

Fancy Nateku Megiri

Luccie


License

This project is licensed under the MIT License.


