# Count_Measure_Objects

Class to measure number of occurences of a repeated object within an image and measure its average size 

## Project Overview

The measurement of the number of occurences of a similar object and its average size within a 2D image is common to several scientific and engineering fields such as 
- metallurgy
- geology
- earth sciences

For example in geology, the grain size is used to study the 
[flow of sediments in river beds](https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1096-9837(199804)23:4%3C345::AID-ESP850%3E3.0.CO;2-B?casa_token=UIcTFPtknAoAAAAA:KBH3_aXOdTmNBRC-7JEqlXBEp0doUwJIJVG4xQbGPDBnkGCyZozqg6xxJBOSWff2bm6U47S6IXZBSw)

In material sciences the grain size distribution of a material will determine their [mechanical properties](https://escholarship.org/content/qt88g8n6f8/qt88g8n6f8.pdf), as is typical in steel and their electrical and optical properties, as is typical in semiconductors such as [silicon](https://www.sciencedirect.com/science/article/abs/pii/S1369800111000886) and [compound semiconductors](https://www.sciencedirect.com/science/article/abs/pii/S1359645413002784)

To determine the size and number of grains per unit volume, usually a determination of this distribution in a 2D image is sufficient if a sensible assumption can be made about the 3D geometry (spherical, columnar, etc.). These 2D images can come from an optical, electronic or any other type of microscope, or from a simple "every-day" digital camera. 

Counting and measuring objects within an image is a technical and in a sense economical problem that is adressed by this package.

## Problem Statement

Several companies offer software that can measure automatically or semi-automatically cluster sizes and distributions.  However they are often limited to the images produced by their metrology equipment, and in most cases they are not free. For Example [Lanoptik](https://www.lanoptik.com/microscope-software-iworks-fg)

Other robust and popular image analysis tools like [ImageJ](https://imagej.net/Welcome/), do not offer to my best knowledge an "operator-free" method to achieve this and be able to measure automatically several images with one command. The problem at hand is to offer a simple class, free of cost, that can count how many clusters (or any self-similar object) are present within in an image and measure their average size. The class is "operator-free" and can be trained to increase the accuracy of the prediction using a fit function.  

Therefore the class Count_Measure_Objects offers 3 main methods

+ Predict: Predicts the number of clusters and their average size
+ Evaluate: Evaluates the accuracy of the model's prediction versus measured images 
+ Fit: trains the classification model with your dataset to optimize the models prediction to your images

## Metrics

The class offers an evaluate_imagefolder method. The metrics used to evaluate the predictions of the model is the root mean square error (rmse) and the coefficient of determination of the cluster number and of the average cluster size. The method can be applied on an image set where the images are labelled as in the following label 1pepper_m_1.6_s_1.5.bmp, where the value after m_ corresponds to the log10(mean size in pixel) and the value after s corresponds to the log10(object counts).

### Installing and deployment

For installation: 

1) Download the folder 60_Grainsize_project and save it on your preffered location. 

2) Run the file Count_Measure_Objects.py to have access to the class Count_Measure_Objects.

3) Instantiate an object of the Count_Measure_Objects class, for example

        my_counter=Count_Measure_Objects()

4) First train and save model with the data given. This will save a checkpoint_vgg11.pth file in the ./DL_pretrained folder. You need to do this only the first time. You can redo this with your own data to tweek the model to your images
        
        my_counter.fit_trainfolder('./DL_data/100_data', 1000)  

5) Use the class methods

+ predict_image: predicts counts and average sizes of objects within an image
+ evaluate_imagefolder :evaluate the prediction versus labelled images
+ fit_trainfolder: fits the classification network to a new set of images

### Users Manual and Methods

The images to be analyzed must be in the .bmp format, they must be of 500x500 in size and have three channels RGB. The class methods are desccribed below

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
            pandas dataframe containing, filname, labelled m and s, predicted m and s
                        class of the image, predicted class
            
            rmse: root mean square errors of the m and s prediction 
            r2: coefficient of determination of m and s predictions

#### fit_trainfolder(datafolder, epochs)
        Args: Datafolder containing the images to trian the network
                The Data folder should contain two folders, train and test
                within train and test the images should be within folders having
                the names 10, 100, and 1000 corresponding to the classes
                
                epochs is the number of epochs to train 
                
        Returns a pandas dataframe with the train and test losses
                as a function of the epochs. Furhtermore a new checkpoint is saved
                under ./DL_pretrained

### Examples

Here are some examples of the main functionalities

First run the file Count_Measure_Objects.py to have access to the Count_Measure_Objects class and be sure that you are working in the ./60_Grainsize_project directory


+ Predict an image
        my_counter=Count_Measure_Objects()
        my_counter.predict_image('./DL_data/200_predict/beans6_m_1.4_s_1.8.bmp')  
        
+ Evaluate the metrics of the model on a measured dataset of class'100' images
        my_counter=Count_Measure_Objects()
        my_counter.evaluate_imagefolder('./DL_data/100_data/train/100','100')          

+ Train 1000 epochs of the neural network to classify your images into 10,100, or 1000 classes
        my_counter=Count_Measure_Objects()
        my_counter.evaluate_imagefolderfit_trainfolder('./DL_data/100_data', 1000)  
       

## Authors

* **Humberto Rodriguez-Alvarez** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This project was done as the capstone project of the Data Scientist program at Udacity. 
