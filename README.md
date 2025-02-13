# CS5567 Final Project: Face Recognition with Transfer Learning  

## Description  
This repository contains the final project for **CS5567**, where a face recognition system is developed using the **Georgia Tech face database** and the **VGG19 pre-trained CNN architecture**. The system employs **transfer learning** and **fine-tuning** to classify **50 subjects**, with verification tasks evaluated using **ROC curves**, **ROC AUC**, and **d-prime metrics**. Additionally, **Rank-1** and **Rank-5** identification rates are analyzed before and after applying a matching threshold.  

## Files Included  

### **Project Code**  
- **File:** `CS5567_DeepLearning_finalProject.m`  
  - **Purpose:** Implements face recognition using transfer learning and fine-tuning of VGG19.  
  - **Key Features:**  
    - Loads and processes face data from the **Georgia Tech dataset**.  
    - Splits data into training, validation, and testing sets under **subject-dependent** and **subject-independent** protocols.  
    - Fine-tunes VGG19 for face recognition.  
    - Computes **ROC curves**, **AUC**, and **d-prime metrics**.  
    - Evaluates **Rank-1** and **Rank-5** identification rates.  

### **Presentation and Documentation**  
- **File:** `CS5567_finalProject.pdf`  
  - Detailed report summarizing the methodology, results, and findings.  
- **File:** `CS5567_FinalProject.pptx`  
  - Slide deck presenting key findings, results, and model performance.  

### **Dataset and Additional Files**  
- **Folders:** `cropped_faces`, `labels`  
  - Contain processed images and label files for training and testing.  

## Installation  
### **Required Dependencies**  
Ensure that **MATLAB** is installed with the following toolboxes:  
- **Deep Learning Toolbox**  
- **Image Processing Toolbox**  
- **Statistics and Machine Learning Toolbox**  

### **Downloading the Dataset**  
The **Georgia Tech Face Database** is included in the repository but can be downloaded from:  
[http://www.anefian.com/research/face_reco.htm](http://www.anefian.com/research/face_reco.htm)  

Once downloaded, unzip the following files into the respective folders:  
- `GTdb_crop.zip` → `cropped_faces/`  
- `labels_gt.zip` → `labels/`  

## Usage  
1. **Open MATLAB** and navigate to the project directory.  
2. Ensure the dataset is extracted into the correct folders.  
3. Run the main script:  
   ```matlab
   CS5567_DeepLearning_finalProject
