# Single-Modality and Multi-Modality Brain Tissue Segmentation   
==============================================================   

## Purpose   
The purpose of this project is to segment brain tissues into white matter (WM), gray matter (GM), and cerebrospinal fluid (CSF) from MR images. A FastSurfer implementation for single-modality segmentation is also available for the same purpose in the directory FastSurfer.

## Environment   
This code was tested in the following environment:  
torch.__version__              = 1.0.1   
torch.version.cuda             = 10.0   
torch.backends.cudnn.version() = 7401   

## Dataset    
Dataset from iSeg 2017 training must be under the folder: iSeg-2017-Training   
Data files (*.hdr and *img) should be directly in the root of the iSeg-2017-Training" folder.   

Dataset from iSeg 2019 testing must be under the folder: iSeg-2019-Testing   
Data files (*.hdr and *img) should be directly in the root of the "iSeg-2019-Testing" folder.   

## Preparing HDF5 files for training.   
Run the following command to prepare the h5 files for training and validation:   
 ```python generate_h5.py ```    

## Training   
For training run the following command:   
```python train.py ``` 

## Validation   
For validation:    
```python validate.py ```    

## Testing   
```python test.py```   

## For FastSurfer implementation:   
Please see ReadMe.md inside FastSurfer directory.   
