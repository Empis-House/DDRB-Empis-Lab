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

import scripts.Grammar_Notation as gn
import scripts.Word_Stitching as ws
import scripts.Performance_Evaluation as pe
import scripts.Map_Grammars_Elite as me
import pandas as pd
import time
import numpy as np
import itertools
import numpy as np
from scipy.stats import linregress

def calculate_slope(Level,df):
    sequence = [list(df.loc[df["token"].astype(int) == Key,"Landings"])[0] for Key in Level]
    # Generate an array of indices as the x-values
    x_values = np.arange(len(sequence))
    
    # Perform linear regression using scipy's linregress
    slope, intercept, r_value, p_value, std_err = linregress(x_values, sequence)
    
    return 1 / (1 + np.exp(-slope)) # mapping it on (0,1) 


# Variety Dominess
def Landings_Score(Level,df):
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

def Colliders_Score(Level,df):
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