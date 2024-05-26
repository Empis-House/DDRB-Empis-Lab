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
    for Key in list(Level):
        value = list(df.loc[df["Key"] == Key,"Landings" ])[0]
        Count += value
        if value>Max:
            Max = value
    if Max == 0:
        print(Level)
        return
        
    return Count/(Max * len(Level))



example_code = ['4-1']
Level_Len = 80
module = 80

#read structures
df = pd.read_csv(r'Super_Mario_Brothers_Maps/structures/Structures_{}.txt'.format(example_code))

#pass info to numpy
Columns = df.columns.to_numpy()
dfnp = np.transpose(df.to_numpy())
Knowledge = dfnp[3:]
Map_Performance= me.Map_Elite(pe.Performance,alphas=[0,0.1,1], Variety_Dominess = [Landings_Score], Grid_points =[[0,0.5]])
Map_Performance.Generate_Mapping(example_code, Knowledge, df, Level_Len)
Map_Performance.Show_Coordinate((1,),Knowledge, df) #Use "level_name" to Save