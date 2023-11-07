# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:34:36 2023

@author: Caeta
"""
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from Class_working_data import Datapreparation
from Class_callsback import Mycallback
import matplotlib.pyplot as plt



class Model(Datapreparation):
    def __init__(self, data_preparation):
        self.data_preparation = data_preparation
        self.inception_v3 = None
        
        
    def inception(self):
        self.inception_v3 = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False,
            input_tensor=None,
            input_shape=(150, 150, 3),
        )  
        # Make all the layers in the pre-trained model non-trainable
        for layer in self.inception_v3.layers:
            layer.trainable = False
            
        self.inception_v3.summary()
        
    def create_model(self):
        
        flat = tf.keras.layers.Flatten()(self.inception_v3.output)
        dropout_layer_1 = tf.keras.layers.Dropout(0.2)(flat)                  
        dense_layer1 = tf.keras.layers.Dense(256, activation='relu')(dropout_layer_1)
        dropout_layer_2 = tf.keras.layers.Dropout(0.2)(dense_layer1)     
        dense_layer2 = tf.keras.layers.Dense(128, activation='relu')(dropout_layer_2)  
        dense_layer3 = tf.keras.layers.Dense(2, activation='softmax')(dense_layer2)           
        
        self.model = tf.keras.models.Model([self.inception_v3.input], dense_layer3) 
        
        self.model.summary()

        self.model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), 
                      loss = 'binary_crossentropy', 
                      metrics = ['accuracy'])
        
    def train_model(self, train_gen, val_gen): 
        train_gen = self.data_preparation.train_gen
        val_gen = self.data_preparation.val_gen
        callbacks = Mycallback()
        history = self.model.fit(
            train_gen,
            batch_size=64,
            validation_data=val_gen,
            epochs=10,
            callbacks = [callbacks]
            )
        
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs = range(len(accuracy))
    
        plt.plot(epochs, accuracy, 'r', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)
        plt.figure()
        plt.show()
        
                   
        
        

        
        