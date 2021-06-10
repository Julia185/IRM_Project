# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:13:20 2021

@author: Julia
"""

import tensorflow as tf
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPool2D

class CNNModel :
    def __init__(self, input_shape) :
        self.__name = "Cnn Model"
        self.__input_shape = input_shape
        # Building a linear stack of layers with the sequential model.
        self.__model = tf.keras.Sequential()
        self.build_model()
        
        
    def build_model(self, features): # VERIFIER LES PARAMETRES DE TOUTES LES FONCTIONS !
        # Convolutional layer.
        # WE CAN ADD MORE CONVOLUTIONAL LAYER.
        self.__model.add(Conv2D(filters = 25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=self.__input_shape))
        self.__model.add(MaxPool2D(pool_size=(1,1), strides = 2))
        
        # Flatten output of convolutional layer.
        self.__model.add(Flatten())
        
        # Hidden Layer.
        self.__model.add(Dense(100, input_shape=self.__input_shape, activation='relu'))
        self.__model.add(Dropout(0.4))
        self.__model.add(Dense(250, activation='relu'))
        self.__model.add(Dropout(0.3))
        
        # Output layer.
        self.__model.add(Dense(10, activation='softmax'))
        
        # Compiling the sequential model.
        self.__model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        
        # Looking at the model summary.
        self.__model.summary()
    
    
    def optimize(self):
        print("Optimizer!")    
        
    def train_Model(self, X_train, Y_train, X_test, Y_test) : # VERIFIER LES PARAMETRES DE TOUTES LES FONCTIONS !!!
        print("Train function")
        # Training the model for 10 epochs.
        self.__model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
        
        
    # reshape fucntion the input. 
    # input = tf.reshape(tensor = features["x"],shape =[-1, 28, 28, 1])
    # first Convolutional Layer
    # conv1 = tf.layers.conv2d(inputs=input,filters=14,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    # Compiling the sequential model.
    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    