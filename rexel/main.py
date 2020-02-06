#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 22:12:13 2020

@author: yefangon
"""

import pickle
import pandas as pd
import numpy as np

#Load stored model:
model = pickle.load(open(r'bestModel.pkl', 'rb'))

# Load data
data = pd.read_csv('data/validation.csv', na_values=[" "])

# Imputing NA's
data['HOUSE'] = data['HOUSE'].fillna(data['HOUSE'].median())
data['LESSTHAN600k'] = np.where(data['HOUSE'] < 600000, 'True', 'False')

#Converting data
data['CUSTOMER_ID'] = pd.Categorical(data['CUSTOMER_ID'])
data['COLLEGE'] = pd.Categorical(data['COLLEGE'])
data['LESSTHAN600k'] = pd.Categorical(data['LESSTHAN600k'])
data['JOB_CLASS'] = pd.Categorical(data['JOB_CLASS'])
data['REPORTED_SATISFACTION'] = pd.Categorical(data['REPORTED_SATISFACTION'])
data['REPORTED_USAGE_LEVEL'] = pd.Categorical(data['REPORTED_USAGE_LEVEL'])
data['CONSIDERING_CHANGE_OF_PLAN'] = pd.Categorical(data['CONSIDERING_CHANGE_OF_PLAN'])

indices = data.index.values

dataset = data.copy()
del dataset['CUSTOMER_ID']

#categorical data encoding (one hot)
predictors = pd.get_dummies(dataset)

#Retain only values
X    = predictors.values

output = data.copy()

#Predictions probabilities
predictProba = model.predict_proba(X)

# 0=STAY and 1=LEAVE
output.loc[indices,'CHURN_PROBABILITY_0'] = predictProba[:,0]
output.loc[indices,'CHURN_PROBABILITY_1'] = predictProba[:,1]

output['CHURN_LABEL'] = np.where(output['CHURN_PROBABILITY_1'] >= 0.5, 'LEAVE', 'STAY')

output['CHURN_PROBABILITY'] = np.where(output['CHURN_LABEL']=='LEAVE',round(output['CHURN_PROBABILITY_1'],2), round(output['CHURN_PROBABILITY_0'],2))

output['CLIENT_TO_CONTACT'] = np.where(output['CHURN_LABEL']=='LEAVE', 'YES', 'NO')

#Discount calculation
output['DISCOUNT'] = np.where(output['CHURN_LABEL']=='LEAVE', round((output['OVERCHARGE']-10)*output['CHURN_PROBABILITY'],2), 0)

#There can't be negatives discounts
output['DISCOUNT'] = np.where(output['DISCOUNT']<0,0,output['DISCOUNT'])

output = output[['CUSTOMER_ID', 'CHURN_PROBABILITY', 'CHURN_LABEL', 'CLIENT_TO_CONTACT', 'DISCOUNT']]

output.to_csv(r'output.csv', index = None, header=True)
