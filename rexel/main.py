#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 22:12:13 2020

@author: yefangon
"""

import pickel
import pandas as pd
import numpy as np

#Load stored model:
model = pickel.load('modelLR.pkl')

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
data['CHURNED'] = pd.Categorical(data['CHURNED'])

dataset = data.copy()
del dataset['CUSTOMER_ID']

#categorical data encoding (one hot)
predictors = pd.get_dummies(dataset)

#Retain only values
X    = predictors.values

# Predictions labels: 1=LEAVE, 0=STAY
predictions      = model.predict(X)
data['CHURNED_LABEL'] = np.where(predictions==1, 'LEAVE', 'STAY')

#Predictions probabilities
data['CHURN_PROBAILITY'] = model.predict_proba(X)

data['CLIENT_TO_CONTACT'] = np.where(data['CHURNED_LABEL']=='LEAVE', 'YES', 'NO')


#Discount calculation
data['DISCOUNT'] = np.where(data['CHURNED_LABEL']=='LEAVE', (data['OVERCHARGE']-10)*data['CHURN_PROBAILITY'], 0)
