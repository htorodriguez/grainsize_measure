# grainsize_measure

Package to measure number of occurences of a repeated object within an image and its average size 

## Project Overview

The measurement of the number of occurences of a similar object and their average size within a 2D image is common to several scientific and engineering fields such as 
- metallurgy
- geology
- earth sciences

For example in geology, the grain size is used to study the 
[flow of sediments in river beds](https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1096-9837(199804)23:4%3C345::AID-ESP850%3E3.0.CO;2-B?casa_token=UIcTFPtknAoAAAAA:KBH3_aXOdTmNBRC-7JEqlXBEp0doUwJIJVG4xQbGPDBnkGCyZozqg6xxJBOSWff2bm6U47S6IXZBSw)

In metallurgy, and more generally in material sciences, the grain size distribution of a material will determine their [mechanical properties](https://escholarship.org/content/qt88g8n6f8/qt88g8n6f8.pdf), as is typical in steel

and their electrical and optical properties, as is typical in semiconductors such as [silicon](https://www.sciencedirect.com/science/article/abs/pii/S1369800111000886) 
and [compound semiconductors](https://www.sciencedirect.com/science/article/abs/pii/S1359645413002784)

To determine the size and number of grains per unit volume, usually a determination of this distribution in a 2D image is sufficient if a sensible assumption can be made about the 3D geometry (spherical, columnar, etc.). These 2D images can come from an optical, electronic or any other type of microscope, or from a simple "every-day" digital camera. 

Counting and measuring mean sizes of repetitive objects within an image is a technical and in a sense economical problem that is adressed by this package. In the following instead of calling them "grains" they will be called "clusters" since in the general case they do not necesarilly have to be grains.

## Problem Statement

Several companies offer software that can measure automatically or semi-automatically cluster sizes and distributions.  However they are often limited to the images produced by their metrology equipment, and they are in most cases not free of cost. For Example [Lanoptik](https://www.lanoptik.com/microscope-software-iworks-fg)

Other quite robust and popular image analysis tools like [ImageJ](https://imagej.net/Welcome/), do not offer to my best knowledge an "operator-free" method to achieve this and be able to measure automatically thousands of images that have different lighting and varying types of clusters. 

The problem at hand is to offer a simple package, free of cost, that can count how many clusters (or any self-similar object) are present within in an image and measure their average size. This should work "operator-free" and can be trained to increase the accuracy of the prediction using a fit function with one single image.  

I emphasize the single-image-training function, since I argue now a days you could easily solve this problem if you had enough labelled examples: for examples thousands of labelled images with all possbile cluster numbers. Current transfer learning methods of available convolutional deep neural networks should be able to achieve this. In fact this is the first method used by this package. 

However, actually getting the labels of so many images is the current task at hand, and this is the task that these package attempts to tackle   

Therefore the package offers 3 main functions

+ Predict: Predicts the number of clusters and their average size
+ Fit: trains the model with one image to optimize the models prediction to this type of image
+ Evaluate: Evaluates the accuracy of the model's prediction versus a known image 

## Metrics

The package offers an evaluate method. The metrics used to evaluate the predictions of the model is the root mean square error (rmse) and the coefficient of determination of the cluster number and of the average cluster size. This can be evaluated on any given test image or set. For demonstration purposes this will be evaluated on a toy data set created by myself by taking pictures of rice and analysing them previously using [imageJ](https://imagej.net)

The coefficient of determination is defined as [explained in wikipedia](https://en.wikipedia.org/wiki/Coefficient_of_determination)

### Installing



## Deployment

This package is available for download under XXXXXXXXXX


### Users Manual

The images to be analyzed must be in the .bmp format, they must be of 500x500 in size and have three channels RGB

The prediction is a tuple of (the number of clusters,  and their average size as a percentage of the image length) 

The package has the following functions

#### predict: 
Input: path to image
Output: tuple of preduction results

#### predict_folder: 
Input: path to image folder
Output: list of tuples containing the image name, 

#### fit: 
Input: tuple of (path to image, (number of clusters, average size as a percentage of the image length ))  
Output: none

Note that the package comes with a trained model, therefore it is not necessary to fit the model to obtain a preduction

#### evaluate:

Input: path to image. The image name must have the format "imagename_m_x.x_s_x.x.bmp", where :
               x.x after m_ is a decimal number equal to the log10(average size in pixels), for example m_1.6
               x.x after s_ is a decimal number equal to the log10(number of clusters), for example s_6.8

Output: tuple of (predicted cluster nr, 
                  predicted average size, 
                  root mean square error of cluster number,
                  root mean square error of cluster size) 

#### evaluate_folder:
Input: path to 500x500 large RGB image folder. The image file names in the folder must have the format
                "imagename_m_x.x_s_x.x.bmp", where :
               x.x after m_ is a decimal number equal to the log10(average size in pixels), for example m_1.6
               x.x after s_ is a decimal number equal to the log10(number of clusters), for example s_6.8
        
Output: tuple of (predicted cluster nr, 
                  predicted average size, 
                  root mean square error of cluster number,
                  root mean square error of cluster size,
                  coefficient of determination R2 of the cluster number prediction,
                   coefficient of determination R2 of the size prediction
                          ) 


## Authors

* **Humberto Rodriguze-Alvarez** 

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This project was done as the capstone project of the Data Scientist program at Udacity. 
