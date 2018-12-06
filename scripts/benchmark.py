import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common_functions import standardize, pre_process_data, train_model,make_country_sub

# data directory
#DATA_DIR = os.path.join('C:\\Datos\\4_Competencias\\3_DrivenData\\predicting_poverty', 'raw_data')
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
According to the graphs looks like:
- Country A is developed country as the number of poor and non-poor are similar.
- Country B is the poorest country as the difference is huge.
- Country C is also a poor country, so it could be in the same category as B.
"""
plt.figure();a_train.poor.value_counts().plot.bar(title='Number of Poor for country A')
plt.figure();b_train.poor.value_counts().plot.bar(title='Number of Poor for country B')
plt.figure();c_train.poor.value_counts().plot.bar(title='Number of Poor for country C')
plt.show()

"""
Country A -> 8203 rows and 345 columns
Country B -> 3255 rows and 442 columns
Country C -> 6469 rows and 164 columns
"""
a_train.info()
b_train.info()
c_train.info()

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
    # Test to see the change in numeric columns 'nEsgxvAq', 'OMtioXZZ', 'YFMZwKrU', 'TiwRslOh'
    numeric = a_train.select_dtypes(include=['int64', 'float64'])
    plt.figure();plt.hist(a_train[numeric.columns[2]])
    plt.figure();plt.hist(aX_train[numeric.columns[2]])
    plt.show()

bX_train = pre_process_data(b_train.drop('poor', axis=1))
by_train = np.ravel(b_train.poor)

cX_train = pre_process_data(c_train.drop('poor', axis=1))
cy_train = np.ravel(c_train.poor)


"""
Train the model using RandomForestClassifier from sklearn with 50 estimators
"""
model_a = train_model(aX_train, ay_train)
model_b = train_model(bX_train, by_train)
model_c = train_model(cX_train, cy_train)


# load test data
a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['C']['test'], index_col='id')

# process the test data
a_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
b_test = pre_process_data(b_test, enforce_cols=bX_train.columns)
c_test = pre_process_data(c_test, enforce_cols=cX_train.columns)


a_preds = model_a.predict_proba(a_test)
b_preds = model_b.predict_proba(b_test)
c_preds = model_c.predict_proba(c_test)

# convert preds to data frames
a_sub = make_country_sub(a_preds, a_test, 'A')
b_sub = make_country_sub(b_preds, b_test, 'B')
c_sub = make_country_sub(c_preds, c_test, 'C')

submission = pd.concat([a_sub, b_sub, c_sub])
submission.head()
submission.tail()

submission.to_csv('submission.csv')
