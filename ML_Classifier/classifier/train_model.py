#data file: https://www.kaggle.com/jacksonchou/hr-data-for-analytics

import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from configs import config


# # Load data and save indices of columns

df = pd.read_csv(config.file_path)
features = df.drop('left', 1).columns

print(f'writing features file at location :{config.feature_pickle}')
pickle.dump(features, open(feature_pickle, 'wb'))

# Fit and save an OneHotEncoder
columns_to_fit = ['sales', 'salary']
enc = OneHotEncoder(sparse=False).fit(df.loc[:, columns_to_fit])

print(f'writing encoder pickle file at location :{config.enc_pickle}')
pickle.dump(enc, open(enc_pickle, 'wb'))

# Transform variables, merge with existing df and keep column names
column_names = enc.get_feature_names(columns_to_fit)
encoded_variables = pd.DataFrame(enc.transform(df.loc[:, columns_to_fit]), columns=column_names)
df = df.drop(columns_to_fit, 1)
df = pd.concat([df, encoded_variables], axis=1)
    
# Fit and save model
X, y = df.drop('left', 1), df.loc[:, 'left']
clf = LGBMClassifier().fit(X, y)

print(f'writing model pickle file at location :{config.mod_pickle}')
pickle.dump(clf, open(mod_pickle, 'wb'))