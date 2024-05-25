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

seed=None
example_code = ['4-1']

Level_Len = 80
module = 80
Game_Mode = 0

#read structures
df = pd.read_csv(r'Super_Mario_Brothers_Maps/structures/Structures_{}.txt'.format(example_code))

#pass info to numpy
Columns = df.columns.to_numpy()
dfnp = np.transpose(df.to_numpy())
Knowledge = dfnp[3:]

# map-elites

Map_Landings_Score = me.Map_Elite(me.Landings_Score,alphas=[0,1])#, Variety_Dominess=[pe.Performance],Grid_points=[[0,0.1,0.25,0.5]])
Map_Performance= me.Map_Elite(pe.Performance,alphas=[0,0.1,1])

level_str = me.Extract_Level_String([example_code[0]],df=df) #+" "+ Extract_Level_String([example_code[1]])
print(me.Extract_Level_String([example_code[0]],df=df))

#Display_Level(Extract_Level_String([example_code[0]]),level_name="1-1".format(example_code),df=df)
#Display_Level(Extract_Level_String([example_code[1]]),level_name="1-2".format(example_code),df=df)

G = gn.Grammar(level_str,knowledge = Knowledge)
pre_Level = G.N_Level_Generator(Level_Len,module=module,Gen_Game_mod=Game_Mode).__repr__()
while not(ws.Jumping_Fiasible_Word(pre_Level, knowledge = Knowledge, Game_mod=Game_Mode)):
    pre_Level = G.N_Level_Generator(Level_Len,module=module,Gen_Game_mod=Game_Mode).__repr__()
Map_Landings_Score.Quest(pre_Level,df=df)
Map_Performance.Quest(pre_Level,df=df)


for j in range(1):
    print("Start",Level_Len)
    print("XX", "t", "Time", "--", "Performance", "--","--","--","--" "Landings_Score", "--","--","--","--", "Fiasible_Word %")
    base_time = time.time()
    t=0
    T=0
    for i in range(1000):
        pre_Level = G.N_Level_Generator(Level_Len,module=module,Gen_Game_mod=Game_Mode).__repr__()
        T+=1
        while not(ws.Jumping_Fiasible_Word(pre_Level, knowledge = Knowledge, Game_mod=Game_Mode)):
            T+=1
            pre_Level = G.N_Level_Generator(Level_Len,module=module,Gen_Game_mod=Game_Mode).__repr__()
        
        t+=1
        Map_Landings_Score.Quest(pre_Level,df=df)
        Map_Performance.Quest(pre_Level,df=df)
        
            
        if i%100 == 0:
            delta_time = time.time() - base_time
            print(i, "t", "%.2f" % (delta_time/60), "--", Map_Performance.Report(), "--","--", Map_Landings_Score.Report(), "--", "%.2f" % (t/T*100),"%")
    delta_time = time.time() - base_time
    print(i, "t", "%.2f" % (delta_time/60), "--", Map_Performance.Report(), "--","--", Map_Landings_Score.Report(), "--", "%.2f" % (t/T*100),"%")
    #Map_Performance.Display_Range(0,3)
    #Map_Landings_Score.Display_Range(0,1)
    

me.Display_Level(Knowledge, Map_Performance.Report(Type = "Strings")[0],level_name="Performance\{}_opt=0".format(example_code),df=df)
me.Display_Level(Knowledge, Map_Performance.Report(Type = "Strings")[1],level_name="Performance\{}_opt=0.1".format(example_code),df=df)  
me.Display_Level(Knowledge, Map_Performance.Report(Type = "Strings")[2],level_name="Performance\{}_opt=1".format(example_code),df=df)


me.Display_Level(Knowledge, Map_Landings_Score.Report(Type = "Strings")[0],level_name="Landings_Score\{}_opt=0".format(example_code),df=df)
me.Display_Level(Knowledge, Map_Landings_Score.Report(Type = "Strings")[1],level_name="Landings_Score\{}_opt=1".format(example_code),df=df)
