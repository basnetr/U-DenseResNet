# Using FastSurfer for IBSR Segmentation   
=========================================

## Dataset   
Download IBSR18 dataset from the website https://www.nitrc.org/projects/ibsr   
The dataset contains MR scans of the brain with manual segmentations.   
The dataset also contains instructions to map different classes of brain into 3 classes (CSF, GM, and GM).   


## Method
FastSurfer is a deep learning based neuroimaging pipeline that can segment whole-brain into 95 classes.   
https://doi.org/10.1016/j.neuroimage.2020.117012   
 
PyTorch implementation for FastSurfer is available at:   
https://github.com/Deep-MI/FastSurfer   


## Running the code
1. Clone the FastSurfer repository and install the requirements to run it.   
2. Copy the following files to ``` FastSurfer/FastSurferCNN ``` directory:   
```preprocess_ibsr.py ```    
This preprocesses the IBSR dataset for FastSurfer's segmentation.   
```predict_ibsr.py```    
This runs the Fastsurfer's segmentation on preprocessed data.   
```postprocess_ibsr.py ```    
This postprocesses the segmentation output and calculates the dice scores for 3 classes of brain tissues (CSF, GM, and GM).   
