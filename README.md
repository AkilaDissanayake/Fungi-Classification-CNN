# 🍄 Fungi Image Classification - PR Group Project

## Project Overview
This repository contains a Jupyter Notebook (PR_Group_Project (2).ipynb) for an image classification project focused on classifying different types of biological samples, specifically fungi (from the dataset defungi.zip).  

The project explores and compares the performance of a *custom-built Convolutional Neural Network (CNN)* against a *state-of-the-art pretrained model (DenseNet121)* under various hyperparameters.

---

## 🎯 Project Goal
The main objective is to accurately classify input images into *five distinct classes*:  
H1, H2, H3, H5, and H6.

---

## 💾 Dataset
The dataset (defungi.zip) consists of images categorized into the five target classes.

- *Source:* Upload to Google Drive under drive/MyDrive/PR_Group_Project/defungi.zip.
- *Preprocessing:*  
  - Resize all images to *32x32 pixels*  
  - Normalize using *ImageNet mean and std*:  
    - Mean: [0.485, 0.456, 0.406]  
    - Std: [0.229, 0.224, 0.225]  
- *Split:*  
  - Training: 70%  
  - Validation: 15%  
  - Testing: 15%  
  - *Random Seed:* 42 for reproducibility  

[Dataset Link](https://archive.ics.uci.edu/dataset/773/defungi)

---

## 💻 Methodology and Experiments

### Models Tested
1. *Model 1: Custom CNN (NormalCNN)*  
   - A custom-defined convolutional neural network.

2. *Model 2: DenseNet121 (Pretrained CNN)*  
   - Uses pretrained DenseNet121 weights for transfer learning.

### Training Parameters
- *Epochs:*  
  - Model 1: 20  
  - Model 2: 3 (fine-tuning)
- *Batch Size:* 64
- *Optimizer:* SGD with Momentum = 0.9
- *Learning Rates Tested:* 0.1, 0.01, 0.001

### Key Observations
- Training can be *time-consuming* (some loops >1 hour).  
- *DenseNet121 achieved:*  
  - Training Accuracy: 85.04%  
  - Validation Accuracy: 81.00% (after 3 epochs)

---

## ⚙ Setup and Dependencies

### Environment
- Recommended to use *GPU environment* (T4 GPU on Colab)
- Ensure the dataset is in:  
  /content/drive/MyDrive/PR_Group_Project/defungi.zip

### Python Dependencies
```bash
pip install numpy pandas scikit-learn tqdm torch torchvision torchaudio
