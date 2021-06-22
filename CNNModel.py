# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:13:20 2021

@author: Julia
"""

import tensorflow as tf
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPool2D, Activation
import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils



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
        
        self.model = self.build_model()


        
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
        #model.summary()
        
        print("Done building model !")
        return model
      
        
    def train_Model(self, X_train, y_train) : 
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
        
        Y_train = np_utils.to_categorical(y_train, 5)

        shuff = list(zip(X_train, Y_train))
        np.random.shuffle(shuff)

        X_train = np.array([shuff[i][0] for i in range(sum(1 for _ in shuff))])
        Y_train = np.array([shuff[i][1] for i in range(sum(1 for _ in shuff))])
        #es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

        # Save model after each epoch to check/bm_epoch#-val_loss
        checkpointer = ModelCheckpoint(filepath="./check/bm_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)

        self.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.n_epoch, validation_split=0.1, verbose=1, callbacks=[checkpointer])
        
        print("Training done !")
        

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
            tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_output, 'model{epoch:08d}.h5'),
                                               mode='auto', monitor='val_loss', verbose=2, save_weights_only=True,
                                               save_best_only=True)
        ]
    
    
    def evaluate(self, x_test, y_test):
        '''
        Compute the loss and accuracy of the model.

        Parameters
        ----------
        x_test : numpy array
        y_test : numpy array

        Returns
        -------
            Returns the test loss.

        '''
        return self.model.evaluate(x_test, y_test, verbose=1)