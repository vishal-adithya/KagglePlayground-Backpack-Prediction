# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 00:26:44 2025

@author: vishaladithyaa

"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv("D:/KagglePlayground-Backpack-Prediction/Data/train.csv")
df.set_index("id",inplace = True)


df.isnull().sum()
df.dropna(how = "any",inplace = True)

print([df["Brand"].unique(),df["Material"].unique(),df["Size"].unique(),df["Style"].unique(),df["Color"].unique()])

def Preprocessing(df):
    brands = df["Brand"].map({"Jansport":0,"Under Armour":1,"Nike":2,"Adidas":3,"Puma":4})
    mats = df["Material"].map({"Leather":0,"Canvas":1,"Nylon":2,"Polyester":3})
    size = df["Size"].map({"Small":0,"Medium":1,"Large":2})
    
    df["Laptop Compartment"] = df["Laptop Compartment"].map(lambda x:0 if x=="No" else 1)
    df["Waterproof"] = df["Waterproof"].map(lambda x:0 if x=="No" else 1)
    
    color = pd.get_dummies(df["Color"],prefix = "Color",dtype=int)
    style = pd.get_dummies(df["Style"],prefix = "Style",dtype=int)
    
    df["Brand"] = brands
    df["Material"] = mats
    df["Size"] = size
    
    df = df.join(color)
    df = df.join(style)
    df.drop(columns = ["Color","Style"],inplace = True)
    
    return df

preprocessed_df = Preprocessing(df=df)  
X = preprocessed_df.drop(columns = ["Price"])
y = df["Price"]
dtrain = xgb.DMatrix(data = X,label=y)
model = xgb.XGBRegressor(objective = "reg:squarederror",device = "gpu",n_estimators = 100,n_jobs = 1,max_depth = 5)
model.fit(X,y)

test_df = pd.read_csv("D:/KagglePlayground-Backpack-Prediction/Data/test.csv")
test_df.set_index("id",inplace = True)
test_df = Preprocessing(test_df)

test_df["Price_Prediction"] = model.predict(test_df)

submission = pd.read_csv("D:/KagglePlayground-Backpack-Prediction/Data/sample_submission.csv")
submission["Price"] = test_df["Price_Prediction"]
submission.set_index("id",inplace = True)
submission.to_csv("submission.csv")
