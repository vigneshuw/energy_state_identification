# State identification of a 5-axis ultra-precision CNC machine tool using energy consumption data assisted by multi-output densely connected 1D-CNN model

This repository contains the code base that was used for the study conducted in the paper. The file ecosystem primarily contains Jupyter Notebook that was used in the model development process, along with the evaluation of the developed models.

This study was conducted at the University of Wisconsin-Madison. For more details please visit [MINLab](https://min.me.wisc.edu/) and/or [Smart Manufacturing at UW-Madison](https://smartmfg.me.wisc.edu/)

DOI Link:- 


## Abstract
<p align="justify">
Ultra-precision machine tools are the foundation for ultra-precision manufacturing. In the era of Industry 4.0, monitoring of the machine tool working condition has become important to improve and control the machining quality. In a conventional setting, numerous sensors are installed by retrofitting the machine tool to monitor the equipment condition effectively. This retrofitment could potentially increase the cost of widespread application of Industry 4.0 technologies. On contrast to the method of retrofitting the machine tool, in this work we propose an intelligent monitoring system that utilizes the equipment's power consumption data to assess and determine the equipment states. The work also discusses the development of a G-code interpreter application that was used to develop equipment working status matrix. The G-code interpreter application has the ability to generate the training data, and extract the features of importance for the Deep learning/Machine learning models, from any type of sensor data. The feature extraction process can be customized as per the requirements by providing template functions to the interpreter application. In our case, the extracted features were the three different power components of a 3-phase AC system. A densely connected convolutional neural network with multiple output was then developed to identify the working state of the machine and predict the feedrate simultaneously. The developed model was then optimized for its hyperparameters - model architecture and learning rate. The best scoring model, for a chosen metric, considering both the outputs was selected. The optimized model was able to identify the working component of the machine with an accuracy of ~94\% and was able to predict the feed rate with a standard deviation of 21.900 from the energy consumption data. The overarching goal of the research work is to predict energy consumed, and augment anomaly detection for an ultra-precision CNC machine tool. The work presented here involves the identification of the equipment state and prediction of the equipment feedrate, and it will serve as a precursor.
</p>

## Content Description

- "model_weights" -> Contains the weights of the trained models
  - "axis_detection" -> Weights for the 1D-CNN model used exclusively for axis detection
  - "feed_prediction" -> Weights for the 1D-CNN model used exclusively for feedrate prediction
  - "multi_output_ax-fr" -> Weights for the multi-output model that can detect the axis and predict feedrate simultaneously

- "plots" -> Plots generated after model evaluation

- "utils" -> Utility functions to enable the model development process

- "data_prep.ipynb" -> Parsing the data output from the G-code interpreter application for model development

- "feature_importance.ipynb" -> Feature importance study on the developed model

- "model_evaluation.ipynb" -> Evaluation of the developed models on chosen metrics

- "model_tune_development.ipynb" -> Hyper-parameters optimization for the developed models

- "mutli-prediction_model.ipynb" -> The development process for the model in the paper

- "simple_1D-CNN.ipynb" -> Benchmarking models. To compare and contrast the multi-output model performance with the single output model performance

- "training_data" -> Data after being parsed by the G-code interpreter application. The directory contains .csv files that are categorized by the axis in motion, with respect to a particular feedrate and machine components active at any point in time.

- "model_data" -> The preprocessed data created by the file "data_prep.ipynb". It is a pickled dictionary of numpy arrays containing the model preprocessed model training data.


## Miscellaneous

**Training Data**

- The data used to train the models is not placed in this repository. If there is a need to get the training data and if there are questions on the DAQ process, please reach out to the owner of this repository. Thank you 

  
  

