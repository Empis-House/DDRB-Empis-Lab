# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:00:20 2023
Check List:
    *More Functions
    *Mapping Functions frome Grids
@author: Juan J. Rueda M.
"""
""""""
# import sys
# sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps")

import pandas as pd
import time
import numpy as np
import itertools
from scipy.stats import linregress

def calculate_slope(Level,df):
    sequence = [list(df.loc[df["token"].astype(int) == Key,"Landings"])[0] for Key in Level]
    # Generate an array of indices as the x-values
    x_values = np.arange(len(sequence))
    
    # Perform linear regression using scipy's linregress
    slope, intercept, r_value, p_value, std_err = linregress(x_values, sequence)
    
    return 1 / (1 + np.exp(-slope)) # mapping it on (0,1) 


# Variety Domains
def landings_score(Level,df):
    if len(Level)==0:
        print(Level)
        return
    Count=0
    Max = -1
    for Key in Level:
        try:
            value = list(df.loc[df["token"].astype(int) == Key,"Landings"])[0]
            Count += value
            if value>Max:
                Max = value
        except Exception as e:
            print(Key, type(Key),"      ", e)
    if Max == 0:
        print(Level)
        return
        
    return Count/(Max * len(Level))

def colliders_score(Level,df):
    if len(Level)==0:
        print(Level)
        return
    Count=0
    Max = -1
    for Key in Level:
        try:
            value = list(df.loc[df["token"].astype(int) == Key,"Colliders" ])[0]
            Count += value
            if value>Max:
                Max = value
        except Exception as e:
            print(Key, type(Key),"      ", e)
    if Max == 0:
        print(Level)
        return
        
    return Count/(Max * len(Level))

variety_domains_json = {"landings": landings_score,
                        "slope": calculate_slope,
                        "colliders": colliders_score}