# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 18:50:12 2023

@author: Caeta
"""

import os 
import zipfile
import pandas as pd

class Reader:
    
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_dir = 'train'
        
    
    def extract_zip(self):
        # Verifying if directories exists:
        if not os.path.exists('train'):
            with zipfile.ZipFile(self.train_path, 'r') as file:
                file.extractall()
             
        if not os.path.exists('test1'):
            with zipfile.ZipFile(self.test_path, 'r') as file:
                file.extractall()
         
        
    def create_dataframe(self):
        data = []
        filenames = os.listdir(self.train_dir)
        labels = [filename.split('.')[0] for filename in filenames]
        
        for filename, label in zip(filenames, labels):
            data.append({'filename': filename, 'label': label})
            
        data_frame = pd.DataFrame(data)
        
        return data_frame
    
    
    
    
        
        