import numpy as np
import pandas as pd

from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression

import os 
os.chdir('Recommendation-System')

bank_full = pd.read_csv('data/classification/bank_full_w_dummy_vars.csv')
print(bank_full.head(), bank_full.info())

# X is all rows (:) and columns 18-36 from bank_full 
# It is all of our input data (0 or 1 for the different criteria)
X = bank_full.iloc[:,[i for i in range(18,37)]].values
# y is all rows and column 17 from y
# It is the value representing the decision (yes or no) 
y = bank_full.iloc[:,17].values

LogReg = LogisticRegression().fit(X, y)
print(LogReg.get_params(), '\n')

# Stats for the user array:
#   [housing_loan | credit_in_default | personal_loans | prev_failed_to_subscribe |
#   prev_subscribed | job_management | job_tech | job_entrepreneur | job_bluecollar | 
#   job_unknown | job_retired | job_services | job_self_employed | job_unemployed | 
#   job_maid | job_student | married | single | divorced]    
deniedUser =   [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]] # Will result in Denied output
approvedUser = [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]] # Will result in Approved output

for newUser in (approvedUser, deniedUser):
    prediction = LogReg.predict(newUser)
    print('    Approved') if prediction == 1 else print('    Deinied')
