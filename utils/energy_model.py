#!/usr/bin/env python
# coding: utf-8

import os 
import sys 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras_tuner as kt


def build_1DCNN_model(layers, counts, act_func, input_shape):

    # Ensure that the arguments entered are correct
    assert (isinstance(layers, tuple) and isinstance(counts, tuple) and isinstance(input_shape,
            tuple) and isinstance(act_func, tuple)), "The layers, counts, act_func and input_shape arguments of the " \
                                                     "function should be tuples"
    assert (len(layers) == len(counts)
            ), "The length of the layer and counts arguments must be equal"
    for i in zip(layers, counts):
        assert (isinstance(i[1], tuple)), "The items within counts must be a tuple"
        assert (isinstance(i[0], str)), "The items within the layers argument must be a string"

    # Creating a sequential model
    model = tf.keras.models.Sequential()

    # Checking for a flattening layer before dense layer
    flatten_layer_check = 0

    # Adding layers and filter to the sequential model depending on the arguments
    for index, layer in enumerate(layers):
        # The first layer should contain the input shape
        if index == 0:
            feature_maps, kernel = counts[index]
            model.add(keras.layers.Conv1D(feature_maps, kernel, activation=act_func[0], padding="same",
                                          input_shape=input_shape))
            continue
            
        # If it is a convolution layer
        if layer == "Conv1D":
            feature_maps, kernel = counts[index]
            model.add(keras.layers.Conv1D(feature_maps, kernel, activation=act_func[0], padding='same',
                                          kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
        # For the layer BatchNormalization
        elif layer == "BatchNormalization":
            value = counts[index][0]
            if value is not None:
                raise ValueError("No slot specified for BN layer index")
            model.add(keras.layers.BatchNormalization())
        # For the layer LayerNormalization
        elif layer == "LayerNormalization":
            value = counts[index][0]
            if value is not None:
                raise ValueError("No slot specified for LN layer index")
            model.add(keras.layers.LayerNormalization())
        # For the layer MaxPooling1D
        elif layer == "MaxPooling1D":
            kernel = counts[index][0]
            model.add(keras.layers.MaxPooling1D(kernel))
        # For the layer GlobalAveragePooling1D
        elif layer == "GlobalAveragePooling1D":
            value = counts[index][0]
            if value is not None:
                raise ValueError ("No slot specified for BN layer index")
            model.add(keras.layers.GlobalAveragePooling1D())
        # For the layer Dense
        elif (layer == "Dense") and (index != len(layers) - 1):
            # Check to ensure that you have called Flatten before calling Dense
            if layers[index - 1] == "Flatten" or flatten_layer_check:
                flatten_layer_check = 1
                neurons = counts[index][0]
                model.add(keras.layers.Dense(neurons, activation=act_func[0],
                                             kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
            else:
                sys.stdout.write("Dense layer applied before a flattening layer")
                sys.exit(1)
        # For the Flatten layer
        elif layer == "Flatten":
            value = counts[index][0]
            if value is not None:
                raise ValueError ("No slot specified for Flatten layer index")
            model.add(keras.layers.Flatten())
            flatten_layer_check = 1
        # For the Dropout layer
        elif layer == "Dropout":
            percent = counts[index][0]
            model.add(keras.layers.Dropout(percent))
        # For the layer Dense at the end
        elif (layer == "Dense") and (index == len(layers) - 1):
            classes = counts[index][0]
            model.add(keras.layers.Dense(classes, activation=act_func[1],
                                         kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
        else:
            sys.stdout.write(f'The mentioned layer is not available -- {layer}')
            sys.exit(1)
            
    # The model returned is uncompiled 
    return model


class ModelTuner(kt.HyperModel):
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    # It has to be build
    def build(self, hp):
        # Defining a sequential model
        model = tf.keras.Sequential()
        
        # Add the input layer and keep it fixed
        model.add(tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same", input_shape=self.input_shape))
        
        # The MainUnit that gets repeated
        for i in range(hp.Int("main_units", 3, 10)):
            for j in range(hp.Int("layers", 2, 5)):
                model.add(tf.keras.layers.Conv1D(hp.Choice("feature_units_" + str(i) + "-" + str(j),
                                                           [128, 256, 512, 1024]),
                                                hp.Int("kernel_units_" + str(i) + "-" + str(j), 3, 7, step=2),
                                                 activation="relu", padding="same"))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling1D(hp.Int("maxpool_units_" + str(i), 2, 6, step=2)))
            
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dropout(hp.Float("dropout", 0.1, 0.7, step=0.2)))
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))
        
        adam = tf.keras.optimizers.Adam(lr=hp.Choice('learning_rate', values=[0.0001, 0.001, 0.01]))
        model.compile(loss="sparse_categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
        
        return model





