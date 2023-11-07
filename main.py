# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 18:55:34 2023

@author: Caeta
"""

from Class_working_data import Datapreparation
from Class_model import Model


train_path = 'train.zip'
test_path = 'test1.zip'

data = Datapreparation(train_path, test_path)
data.spliting_data(flag=False)
data.get_image_median(flag=False)
data.show_images(flag=False)

data.image_generator()

model = Model(data)
model.inception()
model.create_model()
model.train_model(train_gen=data.train_gen, val_gen=data.val_gen)
