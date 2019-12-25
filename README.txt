This code was tested in the following environment:
torch.__version__              = 1.0.1
torch.version.cuda             = 10.0
torch.backends.cudnn.version() = 7401

Step 1: Dataset 
	Dataset from iSeg 2017 training must be under the folder: iSeg-2017-Training
	Data files (*.hdr and *img) should be directly in the root of the "iSeg-2017-Training" folder.

	Dataset from iSeg 2019 testing must be under the folder: iSeg-2019-Testing
	Data files (*.hdr and *img) should be directly in the root of the "iSeg-2019-Testing" folder.

Step 2: Preparing HDF5 files for training.
	Run the following command to prepare the h5 files for training and validation:
	python generate_h5.py

Step 3: Training
	For training run the following command:
	python train.py

Step 4: Validation
	For validation:
	python validate.py

Step 5: Testing
	python test.py

