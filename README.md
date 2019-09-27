# Mitacs-Summer-Project
Automatic segmentation of Hepatocellular carcinoma based on Deep learning and Convolutional Neural Networks : This project was done during my Mitacs GRI ,2019 under the supervision of Dr. Eran Ukwatta at University of Guelph.
All codes are written in python and executed in the Graham cluster of Compute Canada via SHARCNET.

The Whole Slide Images (WSI) of the liver and their corresponding masks are firstly downsampled and chopped up into smaller images each of size 256 X 256 pixels.
The generate_slices.py script is used to downsample and slice the WSI images given in SVS format after convertin them into grayscale and
the generate_mask_patches.py is used to downsample and slice their corresponding masks.

The main.py script is used to train the U-net framework and calculate the test accuracy of the trained network.

The rest of the python scripts provide the required functions to the above three scripts for carrying out their operations, for example : 

unet.py provides main.py the U-net framework architecture for carrying out the segmentation on the images.

data_preparation.py helps the main program to convert the images into the required format before feeding to the network.

process_results.py and helpers.py help in saving the training, predicted and expected images in the desired format.

The .sh files are used to submit the scripts to the Graham cluster as jobs to get them executed.
