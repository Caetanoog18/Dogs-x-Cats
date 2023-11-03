# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 18:55:34 2023

@author: Caeta
"""

from Class_working_with_data import Datapreparation
from sklearn.model_selection import train_test_split

train_path = 'train.zip'
test_path = 'test1.zip'

data = Datapreparation(train_path, test_path)
data.spliting_data(flag=True)
data.get_image_median(flag=True)