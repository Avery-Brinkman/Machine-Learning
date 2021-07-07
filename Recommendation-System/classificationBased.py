import numpy as np
import pandas as pd

from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression

import os 
os.chdir('Recommendation-System')

bank_full = pd.read_csv('data/classification/bank_full_w_dummy_vars.csv')

