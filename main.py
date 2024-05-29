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
from scripts.Variety_Dominess import calculate_slope, Landings_Score, Colliders_Score
import pandas as pd
import time
import numpy as np
import itertools
import numpy as np
from scipy.stats import linregress

def transform_to_cartesian(grid_points):
    # Use itertools.product to get the Cartesian product of the input lists
    cartesian_list = list(itertools.product(*grid_points))
    
    return cartesian_list


example_code = ['1-1']
Level_Len = 40
module = Level_Len # Temp: module != Level_Len could generate buged but solvable levels
Grid_points =[[0,0.5],[0,1]]
alphas=[0,0.1,1] # These are the system foundness acumulation points on the interval [0,1]
#read structures
df = pd.read_csv(r'Super_Mario_Brothers_Maps/structures/Universal_Structures.txt') #.format(example_code))

#pass info to numpy
Columns = df.columns.to_numpy()
dfnp = np.transpose(df.to_numpy())
Knowledge = dfnp[3:]

cartesian_list = transform_to_cartesian(Grid_points)
Map_Performance= me.Map_Elite(pe.Performance,alphas=alphas, Variety_Dominess = [Landings_Score,calculate_slope], Grid_points =Grid_points)
Map_Performance.Generate_Mapping(example_code=example_code, Knowledge=Knowledge, df=df, Level_Len=Level_Len,module=module,batch_size=1,epochs=1000)
Map_Performance.Plot_Coordinate( cartesian_list=cartesian_list, level_curves=[1], Knowledge=Knowledge,df=df) # Use "levels_name" to save plots