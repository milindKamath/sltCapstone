Author: Nikunj Kotecha [ nrk1787@rit.edu ]

#####################################################################################
This folder generates 3 different type of features:

1. Resnet50		( PyTorch model )		"resnet50"
2. Alexnet		( PyTorch model )		"alexnet"
3. Optical Flow						"optical_flow"

#####################################################################################
To run the script:
cmd: make generate

#####################################################################################
Checkout the config file "features.yaml" and make changes accordingly for the
features you want to run.

Ensure that the csv file has 'path' column to it which specifies the path of the
video folder. You can use the script "add_path.py" to add this path column.
The script ensure valid paths only. The config file for is "add_path.yaml"

cmd: make add

