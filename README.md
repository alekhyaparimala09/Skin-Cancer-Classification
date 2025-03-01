# Skin Cancer Classification Using Transfer Learning
## Overview
This project utilizes transfer learning with multiple pre-trained CNN architectures to classify skin lesion images as benign or malignant. The script compares the performance of InceptionV3, ResNet50, VGG16, and EfficientNetB0 models on a skin cancer dataset.
### Features
Implementation of four pre-trained models (InceptionV3, ResNet50, VGG16, EfficientNetB0)

Data augmentation for training images

Automatic validation split for model evaluation

Comprehensive model performance metrics and visualizations

Early stopping to prevent overfitting
## Requirements
•	TensorFlow 2.x
•	NumPy
•	Matplotlib
•	scikit-learn
## Google Colab Setup
This project was designed to run in Google Colab with data stored in Google Drive.
Dataset Structure
## The script expects the following data structure:
/content/drive/MyDrive/skin_cancer_data/

![image](https://github.com/user-attachments/assets/c62b0840-eb5f-45a4-8853-227bcace2c37)

## Usage Instructions
1.	Mount your Google Drive in Colab: 
from google.colab import drive
drive.mount('/content/drive')
2.	Ensure your dataset follows the structure mentioned above
3.	Run the script to train and evaluate all models
4.	Results are saved in the final_multimodel directory including: 
o	Trained model files (.h5)
o	Classification reports
o	Confusion matrices
o	Sample prediction visualizations
## Configuration
Configuration
Model Configuration: Modify the MODEL_CONFIGS dictionary to include or exclude specific models based on your requirements.

Data Augmentation: Adjust the data augmentation parameters in the train_datagen definition to fine-tune image transformations and preprocessing.

Hyperparameters: Change the learning rate and other model hyperparameters by modifying the build_model function.

## Performance Optimization Techniques
Data Augmentation: Enhanced training dataset diversity using rotation, shifts, zoom, brightness variation, and horizontal flips

Dropout Layers: Added 50% dropout for regularization to reduce overfitting

Early Stopping: Implemented with patience=3 to halt training when validation loss plateaus

Pre-trained Weights: Utilized ImageNet weights for transfer learning

Model-specific Preprocessing: Applied appropriate preprocessing for each architecture

Validation Split: Used 20% of training data for validation to monitor generalization

## Performance Metrics
For each model, the script generates:
•	Accuracy and loss curves
•	Confusion matrices
•	Classification reports with precision, recall, and F1-scores
•	Visual examples of predictions on test images
## Performance Results
•	ResNet50: 87% accuracy

![image](https://github.com/user-attachments/assets/7b43c419-4f89-4c60-89f5-896bc306ad07)

•	EfficientNetB0: 85% accuracy
•	VGG16: 84% accuracy
•	InceptionV3: 83% accuracy
## Notes
•	All models are initialized with ImageNet weights
•	Base models are frozen (feature extraction approach)
•	Early stopping monitors validation loss with patience=3



