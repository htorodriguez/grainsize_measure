# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:14:45 2020

@author: hto_r
"""

# =============================================================================
# GLobal imports
# =============================================================================
import numpy as np
from PIL import Image, ImageFilter

# =============================================================================
# 
# =============================================================================
def load_func(file, threshold=100, max_filter=3):
    """Function to load a .bmp image and preprocess it
    args:
        file: string with file name that includes a mean , m, and a stadard deviaton s
            The images within the folder must be labelled as in the following label
            1pepper_m_1.6_s_1.5.bmp
            
            where 
            the value after m corresponds to the log10(mean size in pixel)
            the value after s corresponds to the log10(object counts)
    
    returns 
        img_array: preprocessed image array
        m: string extracted from the file name corresponding to the log(mean radius) 
        s: string extracted from the file name corresponding to the log(counts)
    """
    imgfile=file.split('\\')[-1]
    m = imgfile.split('_')[2]
    ss= imgfile.split('_')[4]
    s = ss.split('.bmp')[0]
    
    img = Image.open(file)
   #
    npImage= np.array(Image.open(file).convert('L'))
    #Get brightness range 
    px_min=np.min(npImage)
    px_max=np.max(npImage)
    #Make a Look up table to scale the image values
    LUT=np.zeros(256, dtype=np.uint8)
    LUT[px_min:px_max+1]=np.linspace(start=0,
                                     stop=255,
                                     num=(px_max-px_min)+1,
                                     endpoint=True,
                                     dtype=np.uint8)
    #Apply LUT and save resulting Image
    img = Image.fromarray(LUT[npImage])
    #find edges
    img= img.filter(ImageFilter.FIND_EDGES)
    #binarize
    def pixelProc(intensity):
        """
        simple threshold function 
        """
        th=threshold
        if intensity > th:
            return 255
        else:
            return 0
    
    img = img.split()[0].point(pixelProc)
    
    #dilation
    img= img.filter(ImageFilter.MaxFilter(max_filter))
    
    img_array=np.array(img)
    #activate to see the image after treatmen
    #img.show()
    
    return img_array, m, s
    