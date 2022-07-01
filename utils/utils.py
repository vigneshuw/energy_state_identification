import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt


def confusion_matrix(model, X_test, y_test, class_names):
    # Prediction on test data set -> predicting classes
    y_pred = model.predict_classes(X_test)
    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
    # Normalizing the confusion matrix
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1) [:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=class_names, columns=class_names)
    # Plotting using a heat map
    figure = plt.figure(figsize=(8, 8))
    sns.set(font_scale=2)
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def check_create_dir(*args):
    # Get the directory name
    temp = ""
    for i in args:
        temp = os.path.join(temp, i)

    # Check if exists else create them all
    if os.path.isdir(temp):
        pass
    else:
        os.makedirs(temp)
