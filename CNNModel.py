# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:13:20 2021

@author: Julia
"""

import tensorflow as tf
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPool2D, Activation
import os
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import json


class CNNModel :
    def __init__(self, n_epoch = 10, n_channel = 4, batch_size = 128, loaded_model=False, w_reg=0.01, n_filters=[64,128,128,128], activation = 'relu', kernel_dims = [7,5,5,3], checkpoint_output = "./checkpoints") :
        '''
        A class to compile the CNN Model, save it and analyze our results.

        Parameters
        ----------
        n_epoch : int, optional
            Number of epochs to train the model on. The default is 10.
        n_channel : int, optional
            Number of channels being assessed. The default is 4.
        batch_size : int, optional
            Number of images to train the model on for each batch. The default is 128.
        loaded_model : bool, optional
            True if loading a pre-existing model. The default is False.
        w_reg : float, optional
            value for l1 and l2 regularization. The default is 0.01.
        n_filters : list, optional
            Number of filters for each convolutional layer (4 total). The default is [64,128,128,128].
        activation : string, optional
            Activation to use at each convolutional layer. The default is 'relu'.
        kernel_dims : list, optional
            Dimension of the kernel at each layer (will be a dim[n] x dim[n] square). The default is [7,5,5,3].
        checkpoint_output : string, optional
            Checkpoint used to save the model at different times. The default is './checkpoints'

        Returns
        -------
        None.

        '''
        
        self.name = "CNN Model"
        self.n_epoch = n_epoch
        self.n_channel = n_channel
        self.batch_size = batch_size
        self.loaded_model = loaded_model
        self.w_reg = w_reg
        self.n_filters = n_filters
        self.activation = activation
        self.kernel_dims = kernel_dims
        self.checkpoint_output = checkpoint_output
        
        self.__model = self.build_model()

        #if not self.loaded_model:
            #self.__model = self.build_model()
        #else :
            #existing_model = str(input('Which model should I load? '))
            #self.__model = self.load_model(existing_model)
        
    def build_model(self): 
        '''
        Compiles model with 4 convolutionnal layers.

        Returns
        -------
        model : keras.Sequential
            CNN Model.

        '''
        
        print("Building Model ...")
        
        # Building a linear stack of layers with the sequential model.
        model = tf.keras.Sequential()
        
        # 1st convolutionnal layers.
        model.add(Conv2D(filters = self.n_filters[0], kernel_size=(self.kernel_dims[0], self.kernel_dims[0]), padding = 'valid', input_shape = (33,33,self.n_channel)))
        model.add(Activation(self.activation))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2), strides = (1,1)))
        model.add(Dropout(0.5))
        
        # 2nd convolutionnal layer.
        model.add(Conv2D(filters = self.n_filters[1], kernel_size=(self.kernel_dims[1], self.kernel_dims[1]), activation = self.activation, padding = 'valid'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2), strides = (1,1)))
        model.add(Dropout(0.5))
        
        # 3rd convolutionnal layer.
        model.add(Conv2D(filters = self.n_filters[2], kernel_size=(self.kernel_dims[2], self.kernel_dims[2]), activation = self.activation, padding = 'valid'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2), strides = (1,1)))
        model.add(Dropout(0.5))
        
        # 4th convolutionnal layer.
        model.add(Conv2D(filters = self.n_filters[3], kernel_size=(self.kernel_dims[3], self.kernel_dims[3]), activation = self.activation, padding = 'valid'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2), strides = (1,1)))
        model.add(Dropout(0.5))
        
        # Flatten output of convolutional layer.
        model.add(Flatten())
        model.add(BatchNormalization())
        
        # Output layer.
        model.add(Dense(10, activation='softmax'))
        
        # Build the optimizer.
        sgd = SGD(lr=0.001, decay=0.01, momentum=0.9)
        
        # Compiling the sequential model.
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
        
        # Looking at the model summary.
        model.summary()
        
        print("Done building model !")
        return model
      
        
    def train_Model(self, X_train, Y_train, X_test, Y_test) : 
        '''
        Function to train the CNN model.

        Parameters
        ----------
        X_train : numpy array
            List of MRI image to train on.
        Y_train : numpy array
            List of MRI image to train on.
        X_test : numpy array
            List of MRI image to test the model.
        Y_test : numpy array
            List of MRI image to test the model.

        Returns
        -------
        None.

        '''
    
        print("Training the model...")
        self.__model.fit(x = X_train, y = Y_train, validation_data = (X_test, Y_test), callbacks = self.__get_callbacks(), steps_per_epoch = int(len(X_train) / self.__batch_size), epochs = self.__n_epoch, verbose = 1)
        print("Training done !")
        

    # TO DO !!!
    def __get_callbacks(self):
        '''
        List of callbacks to apply during training function.

        Returns
        -------
        list
            Callbacks.

        '''
        
        return [
            tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', restore_best_weights = True, patience = 50),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(self.__checkpoint_output, 'model{epoch:08d}.h5'),
                                               mode='auto', monitor='val_loss', verbose=2, save_weights_only=True,
                                               save_best_only=True)
        ]
    
    
    def save_model(self, model_name):
        '''
        Saves current model as a json file.

        Parameters
        ----------
        model_name : string
            Name of the current model to save.

        Returns
        -------
        None.

        '''
        
        model = '{}.json'.format(model_name)
        json_string = self.model_comp.to_json()
        with open(model, 'w') as f:
            json.dump(json_string, f)
    
    
    def load_model(self):
        '''
        Load an already existing model.

        Returns
        -------
        None.

        '''
        weights = '{}.hdf5'.format(self.__name)
        self.__model.load_weights(weights)
    
    
    # reshape fucntion the input. 
    # input = tf.reshape(tensor = features["x"],shape =[-1, 28, 28, 1])    