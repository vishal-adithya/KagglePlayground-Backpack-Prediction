# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 00:26:44 2025

@author: vishaladithyaa

"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("D:/KagglePlayground-Backpack-Prediction/Data/train.csv")
df.set_index("id",inplace = True)

df["Brand"]


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

est = xgb.XGBRegressor(objective="reg:squarederror",
                       device = "cuda",
                       n_jobs = 1,
                       predictor = "gpu_predictor",
                       tree_method = "hist",verbosity = 2,booster = "gblinear")

param_grid = {"max_depth":[5,7,9],
              "learning_rate":[0.1,0.01,0.2],
              "colsample_bytree":[0.5,0.7,0.9],
              "gamma":[0,3,6]}

rsv = RandomizedSearchCV(estimator=est, param_distributions=param_grid,verbose=3,n_jobs=1,cv = 10,n_iter = 15,scoring="neg_root_mean_squared_error")    
rsv.fit(X,y)

best_est = rsv.best_estimator_

rsv.best_params_


gsv = GridSearchCV(estimator=est,param_grid=param_grid,verbose=3,n_jobs=1,cv = 10,scoring="neg_root_mean_squared_error")    
gsv.fit(X,y)
gsv.best_params_
