# Two_Stage_Species_Identification_Model
## COS30049 CTIP Project

The Two-Stage Species Identification Model is the AI core or key area of focus in our group's COS30049 CTIP project. This model combines object detection and species classification techniques to identify and classify species from camera trap images. The first stage leverages YOLOv8 for accurate detection of species, while the second stage utilizes a fine-tuned ResNet50 model for species classification. This system is designed to automate wildlife monitoring by processing large volumes of image data efficiently, aiding in the conservation and study of species in natural habitats.

## DATA PREPARATION (YOLOv8)

### Dependencies
1. os
2. random
3. shutil
4. imgaug
5. numpy
6. PIL
7. cv2
8. albumentations
9. glob

### data_collection.py
This Python script is designed to process and organize camera trap images for species classification. It iterates through the dataset located in a specified directory, checks for species folders, and applies several transformations (like flipping, rotating, and adding noise) to augment the images. The script maps similar species names to a standard format and collects up to 50 images per species (if available) to ensure a balanced dataset. The images are then shuffled and copied to output folders, categorized by species, for further use, such as model training or analysis.

### data_augment.py
This Python script is designed to augment images of underepresented species (or classes) to expand the dataset for model training. It utilizes the albumentations library to apply various transformations (such as horizontal flipping, brightness/contrast adjustment, rotation, scaling, and noise addition) to each image. The augmented images are saved in a new directory while preserving the class folder structure. The script processes a predefined list of species and generates two augmented versions for each original image in the class.

### Dataset Annotation and Augmentation (Roboflow Integration)
The annotations, augmentations, and preprocessing for all images in the final dataset are done using Roboflow. Roboflow provides a comprehensive platform to label, augment, and preprocess image datasets for machine learning projects. The final version of the dataset (v10) has been curated and can be accessed directly via the following link:

[Roboflow Dataset v10](https://universe.roboflow.com/cos30049-ctip/detect-species/dataset/10)

This dataset includes:
- Annotations: Precise labels for all species.
- Augmentations: Enhanced dataset with various transformations to improve model robustness.
- Preprocessing: Optimized data ready for training, ensuring consistency and model-ready formatting.

The final dataset is integrated into the YOLOv8's model pipeline for training and evaluation.

## MODEL TRAINING & EVALUATION (YOLOv8)

### Dependencies 
1. Ultralytics YOLO
2. Google Colab
3. PyTorch
4. Roboflow
5. YAML

### YOLOv8_Species_Detection.ipynb
This Jupyter Notebook demonstrates the training and evaluation of a YOLOv8 model for species detection using a dataset from Roboflow. The notebook includes the following steps:

1. Setup: Installs necessary libraries and checks the device (CPU or GPU) for model training.
2. Dataset Preparation: Downloads and prepares the dataset from Roboflow, configuring it for YOLOv8.
3. Model Training: Trains the YOLOv8 model on the dataset for 25 epochs, with image size set to 800x800.
4. Model Evaluation: Evaluates the model's performance and visualizes confusion matrices and results.
5. Inference: Runs inference on the test images, saving the prediction results.
6. Model Deployment: Deploys the trained model for future use.


## DATA PREPARATION (RESNET)

### Dependencies
1. Ultralytics
2. cv2
3. glob
4. matplotlib
5. numpy
6. pathlib
7. google.colab.drive


### YOLOv8_Cropped_ResNet_Data.ipynb
This Jupyter notebook utilizes a YOLOv8 model to run inference on images from a species detection dataset, drawing bounding boxes around detected objects. It then crops the images based on the predictions and stores them for further use in training a ResNet model. The script includes steps for loading a trained YOLOv8 model, performing inference on random test images, and saving the cropped images to designated directories. The final dataset is organized into train, validation, and test sets, ready for use in model training and evaluation.

## MODEL TRAINING & EVALUATION (RESNET)

### Dependencies
1. torch
2. torchvision
3. matplotlib
4. tensorflow
5. sklearn
6. numpy
7. tqdm
8. seaborn
9. google.colab

### ResNet_Species_Classification.ipynb
This Jupyter notebook demonstrates the training and evaluation of a species classification model using ResNet50. The model is fine-tuned for a custom dataset with 64 classes of species. It includes the following key steps:

1. Data Preprocessing: Images are resized and augmented using torchvision transformations, and the dataset is loaded using ImageFolder.
2. Model Training: A pretrained ResNet50 model is loaded and modified for the classification task. The model is trained for 10 epochs, with training and validation loss and accuracy metrics tracked.
3. Model Evaluation: After training, the model is evaluated on a separate test set, and the results are visualized.
4. Predictions and Inference: The model is used to predict species from random test images, and the results are displayed alongside true labels.
Model Saving: The trained model is saved for future inference tasks.

### Resnet_Evaluation.ipynb
This notebook (Resnet_Evaluation.ipynb) is used to evaluate a pre-trained ResNet50 model for species detection. It loads the test dataset, preprocesses the images, and performs evaluation by generating a classification report and confusion matrix. The model's performance is visualized using a heatmap of the confusion matrix. The notebook is designed to run on Google Colab, where the model and dataset are loaded from Google Drive.

## DATABASE CREATION

### Dependencies
1. mysql

### database.js (/database)
This script sets up a MySQL database for storing and managing data related to wildlife monitoring. It creates a semenggoh database and defines four tables: sites, images, species, and predictions. The sites table stores information about monitoring sites, the images table holds images captured at each site along with metadata, the species table stores details about different species, and the predictions table links images to species predictions with confidence scores.


To run use: node database.js

## RUN & UPLOAD PREDICTIONS TO DATABASE

### Dependencies
1. os
2. cv2
3. pytesseract
4. mysql.connector
5. datetime
6. re
7. ultralytics
8. torch
9. torchvision
10. PIL
11. time

## TwoStage_Predictions_DB.py
This Python script processes images of wildlife captured by camera traps, performs object detection using the YOLOv8 model, and classifies species using a pretrained ResNet50 model. It extracts metadata such as date, time, and temperature from the images using OCR (Tesseract), then stores the relevant information (species, site, date, time, temperature) in a MySQL database. Additionally, detected objects are cropped and saved for further analysis. The script handles database connections, performs preprocessing on images, and ensures data integrity while processing multiple images.
