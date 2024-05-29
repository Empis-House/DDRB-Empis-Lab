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



example_code = ['1-1']
Level_Len = 60
module = 60 # Temp: module != Level_Len could generate buged levels

#read structures
df = pd.read_csv(r'Super_Mario_Brothers_Maps/structures/Universal_Structures.txt') #.format(example_code))

#pass info to numpy
Columns = df.columns.to_numpy()
dfnp = np.transpose(df.to_numpy())
Knowledge = dfnp[3:]
Map_Performance= me.Map_Elite(pe.Performance,alphas=[0,0.1,1], Variety_Dominess = [Landings_Score,Colliders_Score], Grid_points =[[0,0.5],[0,0.5]])
Map_Performance.Generate_Mapping(example_code=example_code, Knowledge=Knowledge, df=df, Level_Len=Level_Len,module=module,batch_size=1,epochs=1000)
Map_Performance.Plot_Coordinate( cartesian_list=[(0, 0), (0, 0.5), (0.5, 0), (0.5, 0.5)], level_curves=[1], Knowledge=Knowledge,df=df) # Use "levels_name" to save plots