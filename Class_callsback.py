# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:06:11 2023

@author: Caeta
"""
import tensorflow as tf

class Mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        current_accuracy = logs.get('accuracy')
        print(f"\nEpoch {epoch + 1} - Accuracy: {current_accuracy}")

        if current_accuracy is not None and current_accuracy > 0.99:
            print(f"\nAccuracy is greater than or equal to 0.99 at epoch {epoch + 1}, so stopping training!")
            self.model.stop_training = True
