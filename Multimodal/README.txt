################################################################################
Author: Nikunj Kotecha [ nrk1787@rit.edu ]

Readme file for running the multimodal transformer code.

################################################################################
There are 6 types of models you can run:

	1. Baseline Joeynmt:
        	- Config file						-> "baseline_joeynmt.yaml"
        	- Uses Joeynmt Encoder and Joeynmt Decoder
        	- Standalone transformer with no multimodality
        	- Works only on one feature

	2. Baseline multimodal:
		- Config file						-> "baseline_mult.yaml"
        	- Uses Multimodal Encoder and Joeynmt Decoder
       		- Standalone transformer with no multimodality
        	- Works only on one feature
    
	3. Multimodal with Joeynmt Encoder:
        	- Config file						-> "mult_joeynmtenc.yaml"
        	- Uses Joeynmt Encoder 
       		- Implements multimodality
        	- Works with three features

    	4. Multimodal:
        	- Config file						-> "mult.yaml"
		- Uses Multimodal Encoder
        	- Implements multimodality
        	- Works with three features

	5. Multimodal modified:
		- Config file						-> "mult_modified.yaml"
		- Uses multimodal encoder and has projection layers
		- Implements multimodality
		- Works with three features

	6. Tri fusion:
		- Config file						-> "mult_fusion.yaml"
		- Uses Multimodal encoder
		- Implements trimodal fusion
		- Works with three features

For our paper submission, the following two models were used:
  
5. Multimodal modified
6. Tri fusion

################################################################################
Remember to add 'words' to the top of the list to the vocab file that
is or you will create. This is required for the dataframe as it will
become the column name.

################################################################################
Different features that these models support for GSL dataset:
    
    1. Joeynmt        ->  'joeynmt'
    2. Resnet50       ->  'resnet50'
    3. Alexnet        ->  'alexnet'
    4. OpenPose       ->  'openpose'
    5. Kmeans         ->  'kmeans'
    6. Optical Flow   ->  'optical_flow'


################################################################################
Points to note in config files:

- There are different configs files for each of the models.
  The config file for each dataset (GSL/ASL) are present in each folder
  Each respective folder has train and inference config files

- In each config file, to use different features:
    -- Pass the name of the feature -> use the same names given as above
    -- Pass the path of the featues -> only the parent dir path
        ---- In case of joeynmt: pass the parent dir path where all pickle files are There
        ---- In case of kmeans: pass the parent dir path where all the csv are kept

- For each model, the respective encoder parameters are changing

- Put True if you want to use cuda

- Put True if you want to log into Wandb
    -- make sure to have an account and api key setup on the system

- Set resume training to True if want to resume training for certain experiment.
  Make sure to set the ckpt path accordingly

- Set transfer learning to True if want to transfer learn from certain model.
  Make sure to set the ckpt path accordingly

################################################################################
For collection of Attention Weights and Visualization of it

- Run the inference script using the inference config files

- Currently, attention weights can only be collected over multimodal modified model

- For visulization of these weights into heatmaps there are two options:
	-- Use the jupyter notebook ( visualization.ipynb )

	-- Use the Flask webapp in folder visualization with the respective config
	   files for each dataset
