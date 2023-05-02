import cv2
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import time

# get the start time
st = time.time()

base_path = './archive/real_vs_fake/real-vs-fake/'
image_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip= True,
    brightness_range=(0.5, 1),
    )
train_flow = image_gen.flow_from_directory(
    base_path + 'train/',
    target_size=(256, 256),
    batch_size=64,
    class_mode='binary'
)

valid_flow = image_gen.flow_from_directory(
    base_path + 'valid/',
    target_size=(256, 256),
    batch_size=64,
    class_mode='binary'
)

from tensorflow.keras import models, layers
model = models.Sequential([
    
    layers.Conv2D(32, (3,3), activation='relu', input_shape = (256, 256, 3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(256, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(512, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(512, activation = 'relu'),
    layers.Dense(2, activation= 'softmax'),
    
])

print(model.summary())

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_flow,
    epochs=50,
    batch_size=64,
    verbose=2,
    callbacks=[early_stop],
    validation_data=valid_flow,
)

filename="CNN140k_final"
model.save("/home/BE_Darshan/test/models/"+filename+".h5",save_format='h5')

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')   