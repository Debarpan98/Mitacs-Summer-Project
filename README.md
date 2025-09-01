# Unet segmentation of Hepatocellular Carcinoma
Automatic segmentation of Hepatocellular carcinoma based on Deep learning and Convolutional Neural Networks : This project was done during my Mitacs GRI, 2019 under the supervision of Prof. Eranga Ukwatta at University of Guelph.
All codes are written in python and executed in the Graham cluster of Compute Canada via SHARCNET.

The Whole Slide Images (WSI) of liver were acquired from the PAIP 2019 grand challenge dataset. The images and their corresponding masks are downsampled and resized into smaller images each of size 256 X 256 pixels.
The [generate_slices](generate_slices.py) script is used to downsample and slice the WSI images given in SVS format after converting them into grayscale and the [generate_mask_patches](generate_mask_patches.py) is used to downsample and slice their corresponding masks given in .TIF format.

The [main](main.py) script is used to train the U-net framework and calculate the test accuracy of the trained network.

The rest of the python scripts provide the required functions to the above three scripts for carrying out their operations, for example : 

[unet](unet.py) provides main.py the U-net framework architecture for carrying out the segmentation on the images.

[data_preparation](data_preparation.py) helps the main program to convert the images into the required format before feeding to the network.

[process_results](process_results.py) and [helpers](helpers.py) help in saving the training, predicted and expected images in the desired format.

The .sh files are used to submit the scripts to the Graham cluster as jobs to get them executed.

# CEPS poster

The results of this project were presented at the CEPS poster day at University of Guelph
![poster](<CEPS Poster_Mitacs GRI.jpg>)
