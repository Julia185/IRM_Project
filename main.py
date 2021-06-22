# -*- coding: utf-8 -*-
"""
Created on Tue May 4 17:48:12 2021

@author: Julia
"""

from DataGenerator import DataGenerator
from CNNModel import CNNModel

if __name__ == '__main__':
    path = r"C:\Users\julia\Desktop\UTBM\Cours\Branche\INFO4\TX52\IRM_Project"

    # We create an instance of the class DataGenerator
    data = DataGenerator(path)
    
    size = 128
    
    x, y, test_x, test_y, val_x, val_y, seq_len = data.loadData(size, generate_array= True)
    
    model = CNNModel()
        
    model.train_Model(x, y)
    
    loss, accuracy = model.evaluate(test_x, test_y)
    print('Loss = {}'.format(loss))
    print('Accuracy = {}'.format(accuracy))