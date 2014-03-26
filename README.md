Comprehensive-Sample-Set-Image-Matting
======================================

## Overview of the program

This program is an implementation of the paper "Improving image matting using comprehensive sample sets" by Shahrian & al. and was coded as part of a science project at university. It includes a small graphical interface to analyze the results and observe the behavior of the algorithm.

![](http://dchiffresdnotes.fr/projects/cssmatting/prev1.png)
![](http://dchiffresdnotes.fr/projects/cssmatting/prev2.png)

**Disclaimer** Is is not an official implementation of the paper (an official binary can be found [here](http://www.alphamatting.com/ComprehensiveSampling.zip)), it is only partial and has not been designed with efficiency concerns. It will not give the same results as the official one and therefore cannot be used to compare the efficiency of this algorithm with other image matting algorithms.

## How to make the program work?

### Compilation

You need to have the library OpenCV installed on your computer (version 2.4.8 recommended, older versions not tested but should work properly). Details regarding the installation procedure for each platform won't be given here but are easily accessible online ([here for example](http://docs.opencv.org/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html)).

A Makefile is provided with the code so compilation shouldn't take more work than simply type make in the code directory. Note that you may have to change the lines LIBS = -L/usr/local/lib and INC = -I/usr/local/include/opencv to point to the directory where OpenCV is installed on your computer.

### Usage

The program must be given as a command-line argument the name of the image (including the extension). Input images should be stored in directory input/ and trimaps in directory trimap/ with the exact same name. Usage example : ./cssmatting GT01.png

Note that a nice dataset of images can be found [here](http://www.alphamatting.com/datasets.php).

## How to use the graphical interface?

### Displaying sample sets and best candidates

Once everything has been computed, the program will open three interactive windows that are synchronized together:

• "Input + (F,B)": Shows the input image. Any click on a pixel will show the best (F,B) pair associated with this point. The color of the line joining them gives an indication of the associated alpha value as a continuous variation from blue (0) to red (1).

• "Alpha Matte": Shows the computed alpha matte. Any click on a pixel will be passed on to the two other windows.

• "Sample set": When no pixel has been selected yet it shows the trimap. When a pixel is selected, this window shows its corresponding subregion, and the associated sample set.

Note that pressing any key will exit the program.

### Changing the objective function

Move the slider in window "Input + (F,B)" to change the objective function for the selection of the best (F,B) pair. You can choose to use only the color constraint, the spatial constraint, the least-overlapping constraint or a combination of these. Note that the alpha matte will be updated (this can take some time depending on the size of the unknown zone).

## Brief description of the data structures

#### Class Region

The most important data structure used in this program is the class Region. It represents a subset of a given image by embedding a list of pixel positions (indexed over the main image 'input'). It provides facilities to get access to the barycenter, mean color and variance of the region, easy access to the equivalent binary map and a function to draw itself on an image. Foreground, Background, Unknown region, subregions, and all clusters are instances of this class.

#### Class CandidateSample

This class is designed to represent a candidate sample. It contains the spatial position, color and a pointer to the region where it was extracted. Sample sets for each subregion are stored as lists of instances of CandidateSample.

## Tweaking the parameters

You can tweak some parameters of the algorithm easily by changing values in the file CSSMatting.cpp (towards the beginning). Parameters that can be changed include the number of subregions, the number of clusters for the first subregion, the type of covariance matrix for the EM algorithm, the choice of the objective function that will be used.

