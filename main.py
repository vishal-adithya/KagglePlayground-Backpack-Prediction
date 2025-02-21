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


brands = df["Brand"].map({"Jansport":0,"Under Armour":1,"Nike":2,"Adidas":3,"Puma":4})
df["Brand"] = brands


mats = df["Material"].map({"Leather":0,"Canvas":1,"Nylon":2,"Polyester":3})
df["Material"] = mats


size = df["Size"].map({"Small":0,"Medium":1,"Large":2})
df["Size"] = size



df["Brand"].unique()
df["Material"].unique()
df["Size"].unique()
df["Style"].unique()
df["Color"].unique()


def Preprocessing(dataframe):
    df = dataframe
    

