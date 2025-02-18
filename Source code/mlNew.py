import os
import pandas as pd # Data Manipulation
from collections import Counter # Data Manipulation
from sklearn.preprocessing import LabelEncoder
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from collections import Counter
import warnings
import pickle

df = pd.read_csv('full_data_flightdelay.csv')
df = df.drop_duplicates()

def reduce_mem_usage(df):
    """ Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
      """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype    
        if col_type == 'category':
            continue
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # Convert object types to category
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
df = reduce_mem_usage(df)

airports_df = pd.read_csv('airports_list.csv')
# Merge the DataFrames to add the state information to the departing airport
df = df.merge(airports_df[['DEPARTING_AIRPORT', 'state']], left_on='DEPARTING_AIRPORT', right_on='DEPARTING_AIRPORT', how='left')
# Rename the state column to DEPARTING_STATE for clarity
df.rename(columns={'state': 'DEPARTING_STATE'}, inplace=True)

#Lấy các thuộc tính train model
col = ['DEP_DEL15','DEP_TIME_BLK', 'CARRIER_NAME', 'DEPARTING_STATE', 'SEGMENT_NUMBER', 'PRCP','LONGITUDE']
df = df[col]

#One-hot encoding
variables = ['DEP_TIME_BLK','CARRIER_NAME','DEPARTING_STATE']
df = pd.get_dummies(df, columns=variables, drop_first = True)

X_train, X_test, y_train, y_test = train_test_split(df.drop('DEP_DEL15', axis=1), df['DEP_DEL15'], test_size=0.3, random_state=42)

rus = RandomUnderSampler(random_state=42, sampling_strategy = 'majority')
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

scaler = MinMaxScaler()
X_train_under = scaler.fit_transform(X_train_under)
X_test = scaler.transform(X_test)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.3,
    random_state=42
)
xgb.fit(X_train_under, y_train_under)
y_pred_xgb = xgb.predict(X_test)


pickle.dump(xgb,open('modelNew.pkl','wb'))