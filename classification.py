import numpy as np
import keras
import pandas as pd
import sklearn
import cv2
import os
import pickle
import random
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import load
from os import makedirs
from os import listdir
import shutil
from shutil import copyfile
from random import seed
from random import random
from matplotlib.pyplot import imread
import sys
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

from google.colab import drive
drive.mount('/content/drive')
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
print(y_train.shape)
y_train = encoder.fit_transform(y_train)
print(y_train)


def define_model():
	
	model = VGG16(include_top=False, input_shape=(256, 256, 3))
	
	for layer in model.layers:
		layer.trainable = False
	
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(19, activation='softmax')(class1)
	
	model = Model(inputs=model.inputs, outputs=output)
	
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
def run_test_harness():
	history = model.fit(x_train,y_train, epochs=10, verbose=1)
	
	_, acc = model.evaluate(x_train,y_train, verbose=0)
	print('> %.3f' % (acc * 100.0))
	
run_test_harness()

def define_model():
	
	model = VGG16(include_top=False, input_shape=(256, 256, 3))
	
	for layer in model.layers:
		layer.trainable = False
	
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(19, activation='softmax')(class1)
	
	model = Model(inputs=model.inputs, outputs=output)
	
	opt = adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
def run_test_harness():
	history = model.fit(x_train,y_train, epochs=10, verbose=1)
	
	_, acc = model.evaluate(x_train,y_train, verbose=0)
	print('> %.3f' % (acc * 100.0))
	
run_test_harness()


def define_model():
	
	model = VGG19(include_top=False, input_shape=(256, 256, 3))
	
	for layer in model.layers:
		layer.trainable = False
	
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(19, activation='softmax')(class1)
	
	model = Model(inputs=model.inputs, outputs=output)
	
	opt = adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
def run_test_harness():
	history = model.fit(x_train,y_train, epochs=10, verbose=1)
	
	_, acc = model.evaluate(x_train,y_train, verbose=0)
	print('> %.3f' % (acc * 100.0))
	
run_test_harness()
