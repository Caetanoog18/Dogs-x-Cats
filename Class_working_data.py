# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 19:21:26 2023

@author: Caeta
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from Class_read import Reader



class Datapreparation(Reader):
    
    def __init__(self, train_path, test_path):
        super().__init__(train_path, test_path)
        self.data_frame = None
        
        
    def spliting_data(self, flag):
        self.extract_zip()
        self.data_frame = self.create_dataframe()
        labels = self.data_frame['label']
        
        
        # x_train contain 80% of the original data data_frame and x_temp cointain 20% of the original data
        self.x_train, self.x_temp = train_test_split(self.data_frame, test_size=0.2, stratify=labels, random_state=42)
        
        label_test = self.x_temp['label']
        # x_test contain 50% from x_temp data and x_val 50% of x_temp data
        self.x_test, self.x_val = train_test_split(self.x_temp, test_size=0.5, stratify=label_test, random_state=42)
        
        if flag is True:
            num_dogs_x_train = len(self.x_train[self.x_train['label'] == 'dog'])
            num_cats_x_train = len(self.x_train[self.x_train['label'] == 'dog'])
            
            num_dogs_x_test = len(self.x_test[self.x_test['label'] == 'dog'])
            num_cats_x_test = len(self.x_test[self.x_test['label'] == 'cat'])
            
            num_dogs_x_val = len(self.x_val[self.x_val['label'] == 'dog'])
            num_cats_x_val = len(self.x_val[self.x_val['label'] == 'cat'])
            
            
            print(f'Number of dogs in training data: {num_dogs_x_train}')
            print(f'Number of cats in training data: {num_cats_x_train}')
            print(f'Number of dogs in test data: {num_dogs_x_test}')
            print(f'Number of cats in test data: {num_cats_x_test}')
            print(f'Number of dogs in validation data: {num_dogs_x_val}')
            print(f'Number of cats in validation data: {num_cats_x_val}')
        
        
    def get_image_median(self, flag):
        train_dir = 'train'
        image_paths = [os.path.join(train_dir, filename) for filename in os.listdir(train_dir)]
        image_sizes = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                width, height = img.size
                image_sizes.append((width, height))
        
        mean_width, mean_height = np.mean(image_sizes, axis=0)
        
        print('mean_size:', mean_width, 'X', mean_height) if flag else None
                
        
        
        