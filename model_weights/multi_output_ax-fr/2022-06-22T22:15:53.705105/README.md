# Trained model weights

## Directories

- "training_history" -> Contains the training history information as a python dictionary and is pickled
- "multi-output_KFold-*_model.h5" -> Trained weights of the multi-output model for each of the cross validation folds

## Training information

- **To determine the impact of dataset size on the model performance for the Y-axis**
- Multiple experiment runs of the Y-axis were conducted inorder to increase the number of corresponding data instances
- No oversampling was applied in this scenario
- The support for each axis
	- X -> 15090
	- Y -> 8526
	- Z -> 6570
	- B -> 11373
	- C -> 11364

## Model Evaluation

1. The confusion matrix. 

![ConfusionMatrix](../../../plots/yaxis_multiple-exp/conmat.png) 

2. Feed rate prediction bar charts

![ErrorHistogram](../../../plots/yaxis_multiple-exp/error_histogram_byaxis.png) 