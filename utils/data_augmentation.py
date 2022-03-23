#!/usr/bin/env python
# coding: utf-8

import sys
import copy
import numpy as np


def naive_resampler(data, proportion=1):

    """
    The naive_resampler used a oversampling technique to increase the proportion of your data within a class

    :param data: Should be a dictionary with keys as classes. The values should be a numpy array with 0 dimension being
                    the count of data
    :param proportion: Proportion by which the other classes should be increased
    :return: The data set dict after oversampling. The keys are classes and the values are numpy array
    """

    # Assert the data-type
    assert (isinstance(data, dict)), "The data input to the function should be a dict with classes as unique keys"
    # Assert the proportion
    assert (proportion <= 1), "The proportion should be a fraction"

    # Find the maximum items class
    max_class = None
    max_item_count = 0
    for key, values in data.items():
        # Get the items count
        count = values.shape[0]

        if max_item_count < count:
            max_item_count = count
            max_class = key
    # Printout the maximum count and the class name
    sys.stdout.write(f"The class with maximum items is {max_class} and the corresponding count is {max_item_count}\n")

    # Deploy the resampling technique
    # Random oversampling
    new_data = copy.deepcopy(data)
    for index, (key, values) in enumerate(data.items()):
        # Get the number of times to resample
        curr_count = values.shape[0]
        diff_count = max_item_count - curr_count
        mod_diff_count = int(diff_count * proportion)           # Modify the count wrt to proportion

        # If the difference is 0
        # continue
        if diff_count == 0:
            continue

        # Get the random sample
        samples = np.random.choice(np.arange(curr_count), size=mod_diff_count, replace=True)

        # pick the items
        picked_data = values[samples]

        # Append the selected data points
        new_data[key] = np.append(new_data[key], picked_data, axis=0)

    return new_data

