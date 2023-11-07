# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:06:11 2023

@author: Caeta
"""
import tensorflow as tf

class Mycallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    
    if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:

      # Stop if threshold is met
      print("\nLoss is lower than 0.4 so cancelling training!")
      self.model.stop_training = True