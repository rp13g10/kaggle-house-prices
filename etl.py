'''This file implements the recommendations made following the work done in eda.py
Initial validation of the data provided was performed in eda.py

Folder structure
- eda.py - General exploration of the dataset
- etl.py - Tidying the datset in preparation for modeling
- modeling.py - Evaluate performance of various models
- generate_predictions.py - Apply final model to generate predictions'''

#%% Import modules
import pandas as pd
from eda import load_validated_data

#%% Load in data
train, metadata = load_validated_data('train.csv')

#%% Merge all porch columns into one

#%% Fill numerical fields with 0 (if it makes sense)

#%% Fill numerical fields with the mean (if 0s don't make sense)

#%% Fill categorical fields with the mode

#%% Convert fields with low variance to binary
