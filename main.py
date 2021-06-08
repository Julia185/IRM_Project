# -*- coding: utf-8 -*-
"""
Created on Tue May 4 17:48:12 2021

@author: Julia
"""

from DataGenerator import DataGenerator, LoaderMode

if __name__ == '__main__':
    mode = LoaderMode.IMAGE_SEQUENCE
    path = r"C:\Users\julia\Desktop\UTBM\Cours\Branche\INFO4\TX52\IRM_Project\images\raw"
    
    # We create an instance of the class DataGenerator
    data = DataGenerator(path, mode=LoaderMode.IMAGE)
    
    if mode == LoaderMode.IMAGE:
        print("IMAGE")
        size = 32
        x, y, test_x, test_y, val_x, val_y, seq_len = data.loadData(size, generate_array=False)
    else:
        print("SEQUENCES")
        size = 32
        x, y, test_x, test_y, val_x, val_y, seq_len = data.loadData(size, generate_array=False)