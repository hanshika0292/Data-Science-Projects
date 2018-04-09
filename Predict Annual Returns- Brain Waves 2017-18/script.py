# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 19:39:21 2017

@author: hanshika
"""
# Load and preprocess data

#https://www.hackerearth.com/challenge/competitive/brainwaves-17-1/
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

sub_ids = test['portfolio_id']
test = test.drop('portfolio_id', axis=1)
y = train['return']
train = train.drop(['portfolio_id', 'return'], axis=1)

# remove some  fields  for simplicity
train = train.drop(['start_date', 'creation_date', 'sell_date', 'indicator_code', 'status', 'desk_id'], axis=1)
test = test.drop(['start_date', 'creation_date', 'sell_date', 'indicator_code', 'status', 'desk_id'], axis=1)

# handle missing values 
train['hedge_value'].fillna(False, inplace=True)
test['hedge_value'].fillna(False, inplace=True)

# missing values for numeric fields
train['sold'].fillna(train['sold'].median(), inplace=True)
train['bought'].fillna(train['bought'].median(), inplace=True)
train['libor_rate'].fillna(train['libor_rate'].median(), inplace=True)
test['libor_rate'].fillna(train['libor_rate'].median(), inplace=True)

# encode categorical fields
obj_cols = [x for x in train.columns if train[x].dtype == 'object']
encoder = LabelEncoder()
for x in obj_cols:
    encoder.fit(train[x])
    train[x] = encoder.transform(train[x])
    test[x] = encoder.transform(test[x])
    
    
# Create a RandomForestRegressor model
    
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=7)
scores = cross_val_score(forest_reg, train, y, scoring='r2', cv=5)
print(scores)
print('mean r2:',np.mean(scores))


# Predict on test set and submit

from IPython.display import FileLink

forest_reg = RandomForestRegressor(random_state=7)
forest_reg.fit(train, y)
preds = forest_reg.predict(test)

sub = pd.DataFrame({'portfolio_id': sub_ids, 'return': preds})
filename = 'sub_returns.csv'
sub.to_csv(filename, index=False)
FileLink(filename)  # lb 0.94277
