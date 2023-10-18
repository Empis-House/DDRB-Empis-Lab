# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:00:20 2023

@author: PC
"""
import sys
sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB")

import Grammar_Notation as gn
import Word_Stitching as ws
import Performance_Evaluation as pe
import pandas as pd
import time

seed=10
example_code = "mix"
Level_Len = 20


level_str = "AABABBCDAEAEABABAABAAWSXMYZAAFFAAETEAAETEAASABASAAEETTEAETTEAAHIAAKLAAESEMEZE[[AAWAA"

df = pd.read_csv(r'C:\Users\PC\Documents\GitHub\EMPIS LAB\Structures_{}_1.txt'.format(example_code), delimiter=',')

def Display_Level(Level, level_name=None, df = df):
    
    print("\n",pe.Performance(Level))
    print(Landings_Score(Level))
    print(ws.Jumping_Fiasible_Word(Level))
            
    Display = pd.DataFrame(columns= ["{}".format(i) for i in range(14)])
    for Key in list(Level):
        new_row = list(list(df.loc[df["Key"] == Key,"Structures" ])[0])
        Display.loc[len(Display)] = new_row
        
        
    Display = Display.transpose()
    if level_name != None:
        Display.to_csv(r'C:\Users\PC\Documents\GitHub\EMPIS LAB\{}.txt'.format(level_name), index=False,header=False)
    
    print(Display)

# Variety Dominess
def Landings_Score(Level, z_count = False):
            
    Count=0
    N_zeros = 0
    for Key in list(Level):
        Count += list(df.loc[df["Key"] == Key,"Landings" ])[0]
        if  not(z_count) and list(df.loc[df["Key"] == Key,"Landings" ])[0] != 0:
            N_zeros += 1
        
    if z_count:
        N_zeros = len(Level)
    return Count/N_zeros


G = gn.Grammar(level_str)
Level = G.N_Level_Generator(Level_Len).__repr__()
P = 0
base_time = time.time()
Level_2 = Level
P_2 = 0
Level_3 = Level
P_3 = 0

for i in range(120):
    pre_Level = G.N_Level_Generator(Level_Len).__repr__()
    while not(ws.Jumping_Fiasible_Word(pre_Level)):
        """Check List
        Surrugate FI-2Pop Search"""
        pre_Level = G.N_Level_Generator(Level_Len).__repr__()
    
    
    if pe.Performance(pre_Level) > P:
        P = pe.Performance(pre_Level)
        Level = pre_Level
        delta_time = time.time() - base_time
        print(i, P, "--", P_2, P_3, "t", delta_time)
        
    if pe.Performance(pre_Level)*Landings_Score(pre_Level) > P_2:
        P_2 = pe.Performance(pre_Level)*Landings_Score(pre_Level)
        Level_2 = pre_Level
        delta_time = time.time() - base_time
        print(i, P, "--", P_2, P_3, "t", delta_time)
        
    if Landings_Score(pre_Level) > P_3:
        P_3 = Landings_Score(pre_Level)
        Level_3 = pre_Level
        delta_time = time.time() - base_time
        print(i, P, "--", P_2, P_3, "t", delta_time)

delta_time = time.time() - base_time
print(i, P, "--", P_2, P_3, "t", delta_time)

Display_Level(Level)
Display_Level(Level_2)
Display_Level(Level_3)

