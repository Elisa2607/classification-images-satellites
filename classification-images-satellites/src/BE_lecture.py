# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 08:48:15 2026

@author: ML
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt




train_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_test = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)


train_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)



train_generator_train = train_train.flow_from_directory(
    directory=r"C:\Users\ML\Desktop\G1_SIA2_images_spatiales\output\train",
    target_size=(600, 600),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

train_generator_test = train_test.flow_from_directory(
    directory=r"C:\Users\ML\Desktop\G1_SIA2_images_spatiales\output\test",
    target_size=(600, 600),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

train_generator_val = train_val.flow_from_directory(
    directory=r"C:\Users\ML\Desktop\G1_SIA2_images_spatiales\output\val",
    target_size=(600, 600),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)