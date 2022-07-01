# Trained model weights

## Directories

- "training_history" -> Contains the training history information as a python dictionary and is pickled
- "multi-output_KFold-*_model.h5" -> Trained weights of the multi-output model for each of the cross validation folds

## Training information

- The very first model developed
- Oversampling was employed
  - Leads to overfitting
- The support for each axis
    - X -> 15090
    - Y -> 15090
    - Z -> 15090
    - B -> 15090
    - C -> 15090

## Model Evaluation

1. The confusion matrix. 

![ConfusionMatrix](../../../plots/with-oversampling_minority-class/conmat.png) 

2. Feed rate prediction bar charts

![ErrorHistogram](../../../plots/with-oversampling_minority-class/error_histogram_byaxis.png) 