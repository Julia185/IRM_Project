# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:13:20 2021

@author: Julia
"""

import tensorflow as tf
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPool2D
import os
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2
from keras.optimizers import SGD


class CNNModel :
    def __init__(self, n_epoch=10, n_chan=4, batch_size = 128, loaded_model=False, w_reg=0.01, n_filters=[64,128,128,128], activation = 'relu', kernel_dims = [7,5,5,3]) :
        self.__name = "CNN Model"
        self.__n_epoch = n_epoch
        self.__in_chan = n_chan
        self.__batch_size = batch_size
        self.__loaded_model = loaded_model
        self.__w_reg = w_reg
        self.__n_filters = n_filters
        self.__activation = activation
        self.__kernel_dims = kernel_dims
        
        if not self.loaded_model:
            self.__model = self.build_model()
        else :
            existing_model = str(input('Which model should I load? '))
            self.__model = self.load_model(existing_model)

        
    def build_model(self): 
        '''
        Compiles model with 4 convolutionnal layers.
        '''
        
        print("Building Model ...")
        
        # Building a linear stack of layers with the sequential model.
        model = tf.keras.Sequential()
        
        # 1st convolutionnal layers.
        model.add(Conv2D(filters = self.n_filters[0], kernel_size=(self.kernel_dims[0], self.kernel_dims[0]), padding = 'valid', W_regularizer=l1l2(l1= self.w_reg, l2 = self.w_reg), input_shape = (self.n_chan,33,33)))
        model.add(Activation = self.__activation)
        model.add(BatchNormalization(mode = 0, axis = 1))
        model.add(MaxPool2D(pool_size=(2,2), strides = (1,1)))
        model.add(Dropout(0.5))
        
        # 2nd convolutionnal layer.
        model.add(Conv2D(filters = self.n_filters[1], kernel_size=(self.kernel_dims[1], self.kernel_dims[1]), activation = self.__activation, padding = 'valid', W_regularizer=l1l2(l1= self.w_reg, l2 = self.w_reg)))
        model.add(BatchNormalization(mode = 0, axis = 1))
        model.add(MaxPool2D(pool_size=(2,2), strides = (1,1)))
        model.add(Dropout(0.5))
        
        # 3rd convolutionnal layer.
        model.add(Conv2D(filters = self.n_filters[2], kernel_size=(self.kernel_dims[2], self.kernel_dims[2]), activation = self.__activation, padding = 'valid', W_regularizer=l1l2(l1= self.w_reg, l2 = self.w_reg)))
        model.add(BatchNormalization(mode = 0, axis = 1))
        model.add(MaxPool2D(pool_size=(2,2), strides = (1,1)))
        model.add(Dropout(0.5))
        
        # 4th convolutionnal layer.
        model.add(Conv2D(filters = self.n_filters[3], kernel_size=(self.kernel_dims[3], self.kernel_dims[3]), activation = self.__activation, padding = 'valid', W_regularizer=l1l2(l1= self.w_reg, l2 = self.w_reg)))
        model.add(BatchNormalization(mode = 0, axis = 1))
        model.add(MaxPool2D(pool_size=(2,2), strides = (1,1)))
        model.add(Dropout(0.5))
        
        # Flatten output of convolutional layer.
        model.add(Flatten())
        model.add(BatchNormalization(mode = 0, axis = 1))
        
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
      
        
    
    # TO VERIFY !!!
    def train_Model(self, X_train, Y_train, X_test, Y_test, batch_size=128, epochs=10) : # VERIFIER LES PARAMETRES DE TOUTES LES FONCTIONS !!!
        print("Train function")
        # Training the model for 10 epochs.
        self.__model.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test), callbacks=self.__get_callbacks(), steps_per_epoch=int(len(X_train) / batch_size), epochs = epochs)


    # TO DO !!!
    def __get_callbacks(self):
       # tensor_board = TensorBoard(log_dir=f'C:\\logs\CNN')
       # model_checkpoint = ModelCheckpoint(filepath="./ckpt/MRI-3D.hdf5",
         #       save_best_only=True,  # Only save a model if `loss` has improved.
         #       monitor="accuracy",
          #      verbose=1,
           # )
        
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=50),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(self.__checkpoint_output, 'model{epoch:08d}.h5'),
                                               mode='auto', monitor='val_loss', verbose=2, save_weights_only=True,
                                               save_best_only=True)
        ]

    
    # TO DO !!!
    def save_model(self, model_name):
        return
    
    
    # TO DO !!!
    def load_model(self, model_name):
        return 
    
    
    
    # reshape fucntion the input. 
    # input = tf.reshape(tensor = features["x"],shape =[-1, 28, 28, 1])
    # first Convolutional Layer
    # conv1 = tf.layers.conv2d(inputs=input,filters=14,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    # Compiling the sequential model.
    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    