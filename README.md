# CS5567 Final Project: Face Recognition with VGG19 and Transfer Learning  

## Description  
This repository contains the final project for **CS5567**, focusing on developing a face recognition system using transfer learning with the **VGG19** pre-trained convolutional neural network (CNN). The model is trained on the **Georgia Tech Face Database** and aims to classify **50 subjects** while evaluating its performance using **ROC curves, AUC, d-prime, and rank-based identification rates**.  

## Files Included  

### **Project Implementation**  
- **File:** CS5567_DeepLearning_finalProject.m  
- **Description:**  
  - Loads the **Georgia Tech Face Database** (GTdb) for training.  
  - Implements **transfer learning** with the **VGG19 CNN architecture**.  
  - Performs **data augmentation** (random translation, reflection).  
  - Splits the dataset into **training, validation, and test sets**.  
  - Extracts deep features from the **fc7 layer**.  
  - Computes **cosine similarity for verification**.  
  - Evaluates model performance using **ROC AUC and d-prime metrics**.  
  - Calculates **rank-1 and rank-5 identification rates** before and after threshold filtering.  

### **Project Report**  
- **File:** CS5567_finalProject.pdf  
- **Contents:**  
  - Explanation of dataset preprocessing, training methodology, and evaluation.  
  - Discussion of subject-dependent and subject-independent protocols.  
  - Interpretation of ROC curves, AUC, and d-prime values.  
  - Comparison of different training strategies and fine-tuning techniques.  

### **Presentation Slides**  
- **File:** CS5567_FinalProject.pptx  
- **Contents:**  
  - Visual overview of the project, including sample dataset images.  
  - Performance evaluation plots: ROC, probability density histograms.  
  - Insights on network training, data augmentation, and accuracy improvements.  

### **Dataset Folders**  
- **Folder:** cropped_faces  
  - Contains **750 images** of **50 subjects** (15 images per subject).  
- **Folder:** labels  
  - Stores label information corresponding to the dataset images.  

## Installation  

### Required MATLAB Toolboxes  
- Deep Learning Toolbox  
- Computer Vision Toolbox  

### Dependencies  
The dataset is **included in the repository**, so no additional downloads are required. Ensure that your MATLAB environment supports deep learning with **VGG19**.  

## Usage  

1. **Extract the dataset:**  
   - Ensure the **cropped_faces** and **labels** folders are present in the working directory.  
2. **Run the main script:**  
   - Open **CS5567_DeepLearning_finalProject.m** in MATLAB.  
   - Execute the script to **train and evaluate the model**.  
3. **Analyze results:**  
   - The script generates **ROC curves**, **histograms**, and **performance metrics**.  
   - Outputs **rank-1 and rank-5 identification rates**.  

## Example Output  

- **Verification Performance**  
  - **ROC AUC:** **0.99926** (Near-perfect classification)  
  - **d-prime:** **4.3733** (Strong separation of genuine and imposter distributions)  

- **Identification Accuracy**  
  - **Rank-1 Identification Rate:** **98%**  
  - **Rank-5 Identification Rate:** **100%**  
  - **After Applying a Matching Threshold:**  
    - **Filtered Rank-1:** **98%**  
    - **Filtered Rank-5:** **98%**  

## Contributions  
This project is part of the **CS5567 Deep Learning Course Final Project**. Contributions are welcome for further improvements in transfer learning and dataset augmentation.  

## License  
This project is open for educational and research use.  

---
**Author:** Alexander Dowell  
