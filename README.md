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
+ Evaluate: Evaluates the accuracy of the model's prediction versus measured images 
+ Fit: trains the model with one image to optimize the models prediction to this type of image

## Metrics

The package offers an evaluate method. The metrics used to evaluate the predictions of the model is the root mean square error (rmse) and the coefficient of determination of the cluster number and of the average cluster size. This can be evaluated on an image set where the images are labelled as in the following label 1pepper_m_1.6_s_1.5.bmp where 
the value after m_ corresponds to the log10(mean size in pixel) and the value after s corresponds to the log10(object counts).

### Installing and deployment
For installation download the folder 60_Grainsize_project and save it on your prefered location. Please take into account that a pretrained model (~530MB) is within the folder ./DL_pretrained. If you do not want to download the model, you can download everything else and first train a model on your data using the fit_trainfolder method.

Fun the file grain_size_class.py to have access to the class Count_Measure_Objects.
Finally instantiate an object of the Count_Measure_Objects class, for example

p=Count_Measure_Objects()

The objects of the Count_Measure_Objects class have three methods:

+ predict_image: predicts counts and average sizes of objects within an image
+ evaluate_imagefolder :evaluate the prediction versus labelled images
+ fit_trainfolder: fits the classification network to a new set of images

### Users Manual and Methods

The images to be analyzed must be in the .bmp format, they must be of 500x500 in size and have three channels RGB

#### predict_image(image_path)
        Args: 
            image_path: string with image path     
        Returns: 
            pred_class: int with predicted class, 10, 100, or 1000
            clusters: int with predcited number of clusters
            mean_radius: int with mean radius in pixels
            log_mean_radius: float with log10(mean radius)

#### evaluate_imagefolder(datafolder,folder_class)
        Args: 
            datafolder: Path to a datafolder containing the labelled images
            folder_class:The class of the images in the data folder must be given as a str
            eg: '100'
            
            The images within the folder must be labelled as in the following label
            1pepper_m_1.6_s_1.5.bmp
            
            where 
            the value after m corresponds to the log10(mean size in pixel)
            the value after s corresponds to the log10(object counts)
        
        Returns: 
            dataframe containing, filname, labelled m and s, predicted m and s
                        class of the image, predicted class
            
            rmse: root mean square errors of the m and s prediction 
            r2: coefficient of determination of m and s predictions

#### fit_trainfolder(datafolder, epochs)
        Args: Datafolder containing the images to trian the network
                The Data folder should contain two folders, train and test
                within train and test the images should be within folders having
                the names 10, 100, and 1000 corresponding to the classes
                
                epochs is the number of epochs to train 
                
        Returns: A dataframe with the train and test losses
                as a function of the epochs. A new checkpoint is saved 


## Authors

* **Humberto Rodriguze-Alvarez** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This project was done as the capstone project of the Data Scientist program at Udacity. 
