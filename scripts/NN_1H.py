"""
Planar data classification with one hidden layer
"""
# Package imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from common_functions import standardize,standardize,pre_process_data
from coursera_1_2_functions import sigmoid
# from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from NN_1H_functions import *
np.random.seed(1) # set a seed so that the results are consistent


"""
Data Preparation
"""
# data directory
# DATA_DIR = os.path.join('C:\\Datos\\4_Competencias\\3_DrivenData\\predicting_poverty', 'raw_data')
DATA_DIR = os.path.join('/Users/jprocha/Datos/drivendata/poverttest', 'raw_data')

data_paths = {'A': {'train': os.path.join(DATA_DIR, 'A_hhold_train.csv'),
                    'test':  os.path.join(DATA_DIR, 'A_hhold_test.csv')},
                    'B': {'train': os.path.join(DATA_DIR,'B_hhold_train.csv'),
                    'test':  os.path.join(DATA_DIR,'B_hhold_test.csv')},
                    'C': {'train': os.path.join(DATA_DIR,'C_hhold_train.csv'),
                    'test':  os.path.join(DATA_DIR,'C_hhold_test.csv')}}

# load training data
a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
c_train = pd.read_csv(data_paths['C']['train'], index_col='id')
"""
aX_train -> 8203 rows, 859 columns
ay_train -> 8203 rows
bX_train -> 3255 rows, 1432 columns
by_train -> 3255 rows
cX_train -> 6469 rows, 795 columns
cy_train -> 6469 rows

Description:
    Numeric columns were standarized using Z-score scaling
    Varchar columns were replaced by boolean dummies
"""
aX_train = pre_process_data(a_train.drop('poor', axis=1))
ay_train = np.ravel(a_train.poor)

bX_train = pre_process_data(b_train.drop('poor', axis=1))
by_train = np.ravel(b_train.poor)

cX_train = pre_process_data(c_train.drop('poor', axis=1))
cy_train = np.ravel(c_train.poor)


# load test data
a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['C']['test'], index_col='id')

# process the test data
a_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
b_test = pre_process_data(b_test, enforce_cols=bX_train.columns)
c_test = pre_process_data(c_test, enforce_cols=cX_train.columns)

# Reshape Input records
X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(aX_train, ay_train, test_size = 0.2, random_state=42, stratify=ay_train)
X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(bX_train, by_train, test_size = 0.2, random_state=42, stratify=by_train)
X_C_train, X_C_test, y_C_train, y_C_test = train_test_split(cX_train, cy_train, test_size = 0.2, random_state=42, stratify=cy_train)

X_A_train=X_A_train.T
y_A_train=y_A_train.reshape(1,y_A_train.shape[0])
X_A_test=X_A_test.T
y_A_test=y_A_test.reshape(1,y_A_test.shape[0])

X_B_train=X_B_train.T
y_B_train=y_B_train.reshape(1,y_B_train.shape[0])
X_B_test=X_B_test.T
y_B_test=y_B_test.reshape(1,y_B_test.shape[0])

X_C_train=X_C_train.T
y_C_train=y_C_train.reshape(1,y_C_train.shape[0])
X_C_test=X_C_test.T
y_C_test=y_C_test.reshape(1,y_C_test.shape[0])

"""
Neural Network Process
"""
# Only testing with dataset A
(n_x, n_h, n_y) = layer_sizes(X_A_train, y_A_train)
parameters = initialize_parameters(n_x, n_h, n_y)
A2, cache = forward_propagation(X_A_train,parameters)
print("cost = " + str(compute_cost(A2, y_A_train, parameters)))

grads = backward_propagation(parameters, cache, X_A_train, y_A_train)

parameters = update_parameters(parameters, grads)

parameters = nn_model(X_A_train.values, y_A_train, 4, num_iterations=10000, print_cost=True)
