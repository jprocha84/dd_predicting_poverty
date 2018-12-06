import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from coursera_1_2_functions import sigmoid, initialize_with_zeros, propagate, optimize, predict, model, predict_V2,make_country_sub
from sklearn.model_selection import train_test_split
from common_functions import standardize,standardize,pre_process_data

# data directory
DATA_DIR = os.path.join('C:\\Datos\\4_Competencias\\3_DrivenData\\predicting_poverty', 'raw_data')

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

"""
MODEL TRAINING
"""
X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(aX_train, ay_train, test_size = 0.2, random_state=42, stratify=ay_train)
X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(bX_train, by_train, test_size = 0.2, random_state=42, stratify=by_train)
X_C_train, X_C_test, y_C_train, y_C_test = train_test_split(cX_train, cy_train, test_size = 0.2, random_state=42, stratify=cy_train)

model_a = model(X_A_train.T, y_A_train.reshape(1,y_A_train.shape[0]), X_A_test.T, y_A_test.reshape(1,y_A_test.shape[0]), num_iterations = 3000, learning_rate = 0.06, print_cost = True)
model_b = model(X_B_train.T, y_B_train.reshape(1,y_B_train.shape[0]), X_B_test.T, y_B_test.reshape(1,y_B_test.shape[0]), num_iterations = 3000, learning_rate = 0.06, print_cost = True)
model_c = model(X_C_train.T, y_C_train.reshape(1,y_C_train.shape[0]), X_C_test.T, y_C_test.reshape(1,y_C_test.shape[0]), num_iterations = 3000, learning_rate = 0.01, print_cost = True)


# Plotting Cost
costs = np.squeeze(model_a['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(model_a["learning_rate"]))
plt.show()

# Training with different LEARNING RATES
def checkDiffLearningRates(xTrain,yTrain,xTest,yTest,learningRates, num_iterations):
    learning_rates = learningRates
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = model(xTrain,yTrain,xTest,yTest, num_iterations = num_iterations, learning_rate = i, print_cost = False)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations')

    # The model with Learning Rate 0.01 is performing better in the dev set
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()

checkDiffLearningRates(X_A_train, y_A_train, X_A_test, y_A_test,[0.9, 0.06, 0.01, 0.001, 0.0001], 3000) # Best 0.06
checkDiffLearningRates(X_B_train.T, y_B_train.reshape(1,y_B_train.shape[0]), X_B_test.T, y_B_test.reshape(1,y_B_test.shape[0]),[0.06, 0.01, 0.001, 0.0001], 3000) # Best 0.06
checkDiffLearningRates(X_C_train.T, y_C_train.reshape(1,y_C_train.shape[0]), X_C_test.T, y_C_test.reshape(1,y_C_test.shape[0]),[0.9, 0.06, 0.01, 0.001, 0.0001], 3000) # Best 0.01

# PREDICT in the TEST Set
a_preds = predict_V2(model_a['w'],model_a['b'],a_test.T)
b_preds = predict_V2(model_b['w'],model_b['b'],b_test.T)
c_preds = predict_V2(model_c['w'],model_c['b'],c_test.T)

print(np.shape(a_preds))
print(np.shape(b_preds))
print(np.shape(c_preds))

a_sub = make_country_sub(a_preds.T.values, a_test, 'A')
b_sub = make_country_sub(b_preds.T, b_test, 'B')
c_sub = make_country_sub(c_preds.T, c_test, 'C')

submission = pd.concat([a_sub, b_sub, c_sub])
print(submission.head())
print(submission.tail())

submission.to_csv('submission.csv')
