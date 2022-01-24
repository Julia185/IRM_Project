# -*- coding: utf-8 -*-
"""
Created on Mon May 3 17:10:47 2021

@author: Julia
"""

import cv2
import numpy as np

from DataGenerator import LoaderMode

class ImagePreProcess :
    def __init__(self, mode=LoaderMode.IMAGE):
        self.__mode = mode
    
    def modifyImage(self, train_set, validation_set, test_set):
        '''
        Load the MRI images and use preprocessing function on it.

        Parameters
        ----------
        train_set : set
            Set used for the training part.
        validation_set : set
            Set used for the validation part.
        test_set : set
            Set used for the testing part.

        Returns
        -------
        train_set : set
            Set used for the training part.
        validation_set : set
            Set used for the validation part.
        test_set : set
            Set used for the testing part.

        '''
        
        # For every set, we pre process the images.
        for i in range(train_set.shape[0]) :
            for j in range(train_set.shape[3]):
                train_set[i, :, :, j] = self.imagePreProcess(train_set[i, :, :, j])
        for i in range(validation_set.shape[0]) :
            for j in range(validation_set.shape[3]):
                validation_set[i, :, :, j] = self.imagePreProcess(validation_set[i, :, :, j])
        for i in range(test_set.shape[0]) :
            for j in range(test_set.shape[3]):
                test_set[i, :, :, j] = self.imagePreProcess(test_set[i, :, :, j])
        return train_set, test_set, validation_set



    def imagePreProcess(self,image):
        '''
        Pre processing of the MRI images.

        Parameters
        ----------
        image : image
            Image to .

        Returns
        -------
        dilate : TYPE
            DESCRIPTION.

        '''
        # We save the parameters of the image.
        height, width = image.shape[:2]
    
        # Our image is already in GRAYSCALE since it's a MRI.
        # First, we apply a Gaussian filter that blur the image.
        blur = cv2.GaussianBlur(image, (5,5), 0)
        # Then we save the image.
        cv2.imwrite('Blur.png', blur)
    
        # Then, we need to convert our image to binary number (like MNIST example).
        # We start with a simple Stresholding : Tresh_Binary.
        # We set a threshold value, if the pixel value is smaller, 
        # its number is set to 0 (black), else 255.
        # Verify the threshold number !
        thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)[1]
        # Then we save the image.
        img=np.array(thresh)
        cv2.imwrite('Thresh.png', img)
    
        # We now want to remove any noise in the image.
        # We continue by using the erode function.
        # It shrinks bright regions and enlarges dark regions.
        erod = cv2.erode(np.array(thresh), None, cv2.BORDER_REFLECT)
        # Then we save the image.
        img=np.array(erod)
        cv2.imwrite('Erod.png', erod)
    
        # Finally, we use a dilation function twice to increase our focused area.
        # It shrinks dark regions and enlarges the bright regions.
        dilate = cv2.dilate(erod, None, iterations=2)
        # Then we save the image.
        img=np.array(dilate)
        cv2.imwrite('Dilate.png', dilate)
    
        return dilate
        