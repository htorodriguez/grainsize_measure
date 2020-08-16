# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 10:48:31 2020
Class of the instance counting and measuring algorithm 
@author: hto_r
"""

# =============================================================================
# GLobal imports
# =============================================================================
# =============================================================================
# loal imports
# =============================================================================
import Grain_size_predict
import DL_optimization

# =============================================================================
# 
# =============================================================================

class Count_Measure_Objects():
    """ Class for 
                - predicting counts and sizes of instances within an image
                - fitting the model to a set of images
                - evaluating the model versus a known image 
    Attributes:
                - image_path              
    """
    
    def __init__(self, image_path='./'):
        self.image_path=image_path
 
    def predict_image(self, image_path='./'):   
        """Function to predict counts and image size     
        Args: 
            string with image_path     
        Returns: 
            pred_class: int with predicted class, 10, 100, or 1000
            clusters: int with predcited number of clusters
            mean_radius: int with mean radius in pixels
            log_mean_radius: float with log10(mean radius)
        """
        self.image_path=image_path
        
        (self.pred_class, 
         self.clusters, 
         self.mean_radius, 
         self.log_mean_radius) =Grain_size_predict.grain_size_predict(self.image_path)
        
        return(self.pred_class, self.clusters, self.mean_radius,self.log_mean_radius) 

    def evaluate_imagefolder(self, datafolder,folder_class):
    
        """Function to evaluate several labelled images of the same class
            against the prediction
        
        Args: 
            datafolder: Path to a datafolder containing the labelled images
            folder_class: The class of the images in the data folder must be given as a str
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
        """

        (self.eval_df, 
         self.m_rmse, 
         self.s_rmse, 
         self.m_r2, 
         self.s_r2,
         self.class_acc,) =Grain_size_predict.predict_compare_grain_size(datafolder, 
                                                                      folder_class) 
        print('m_rmse', self.m_rmse)
        print('s_rmse', self.s_rmse)
        print('m_r2', self.m_r2)
        print('s_r2', self.s_r2)
        print('classification_accuracy', self.class_acc)
        return(self.eval_df) 

    def fit_trainfolder(self, datafolder, epochs):
        """Function to train the neural network to a particular set of images     
        Args: Data folder containing the images to trian the network
                The Data folder should contain two folders, train and test
                within train and test the images should be within folders having
                the names 10, 100, and 1000 corresponding to the classes
                
                epochs is the number of epochs to train 
                
        Returns: A dataframe with the train and test losses
                as a function of the epochs. A new checkpoint is saved 
        """   
        self.DL_df=DL_optimization.import_train_vgg(datafolder+'/train',datafolder+'/test', epochs)
            
            
        return(self.DL_df) 