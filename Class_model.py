# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:34:36 2023

@author: Caeta
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from Class_working_data import Datapreparation
from Class_callsback import Mycallback
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report, ConfusionMatrixDisplay


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
        test_gen = self.data_preparation.test_gen
        self.test1_gen = self.data_preparation.test1_gen
        my_callbacks = Mycallback()
        history = self.model.fit(
            train_gen,
            batch_size=64,
            validation_data=val_gen,
            epochs=10,
            callbacks = [my_callbacks]
            )
        
        
        self.plot_training_history(history)
        self.model_evaluation(train_gen, val_gen, test_gen)
        
        
        
    def plot_training_history(self, history):
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.show()
    
    
    def model_evaluation(self, train_gen, val_gen, test_gen):
        
        train_score = self.model.evaluate(train_gen, steps= 32 , verbose = 1)
        val_score = self.model.evaluate(val_gen, steps= 32 , verbose = 1 )
        test_score = self.model.evaluate(test_gen, steps= 32 , verbose = 1 )
    
    
        print(f'Train loss = {train_score[0] }')
        print(f'Train Accuracy = {train_score[1]}')
        print(f'Validation loss = {val_score[0]}')
        print(f'Validation Accuracy = {val_score[1]}')
        print(f'Test loss = {test_score[0]}')
        print(f'Test Accuracy = {test_score[1]}')
        
        predictions = self.model.predict(test_gen)
        
        # Convert prediction to labels
        predicted_labels = [1 if pred[0] > 0.5 else 0 for pred in predictions]
        
        # Get true labels for the test set
        true_labels = test_gen.classes
        
        matrix = confusion_matrix(true_labels, predicted_labels)
        
        # Display the confusion matrix as a heatmap
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=test_gen.class_indices)
        disp.plot(cmap='viridis', values_format='d')
        
        
        # Generate and print the classification report
        classification_rep = classification_report(true_labels, predicted_labels, target_names=test_gen.class_indices)
        print("\nClassification Report:")
        print(classification_rep)
        
    def test_prediction(self):
        test1_predict = np.argmax(self.model.predict(self.test1_gen), axis=1)
        
        label_mapping = {0: 'cat', 1: 'dog'}
        self.data_preparation.test_data['label'] = pd.Series(test1_predict).map(label_mapping)
        
        fig, axes = plt.subplots(1, 10, figsize=(20, 4))
        for idx, (_, row) in enumerate(self.data_preparation.test_data.head(10).iterrows()):
            image_path = os.path.join('test1', row['filename'])
            image = Image.open(image_path)
            axes[idx].imshow(image)
            axes[idx].set_title("Label: " + row['label'])
            axes[idx].axis('off')
        plt.show()


    
        
    