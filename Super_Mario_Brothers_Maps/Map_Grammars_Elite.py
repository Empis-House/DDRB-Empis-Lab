# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:00:20 2023
Check List:
    *More Functions
    *Mapping Functions frome Grids
@author: Juan J. Rueda M.
"""
""""""
import sys
sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps")

import Grammar_Notation as gn
import Word_Stitching as ws
import Performance_Evaluation as pe
import pandas as pd
import time
import numpy as np

seed=10
example_code = "mix"
Level_Len = 20


level_str = "AABABBCDAEAEABABAABAAWSXMYZAAFFAAETEAAETEAASABASAAEETTEAETTEAAHIAAKLAAESEMEZE[[AAWAA"

df = pd.read_csv(r'C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps\Structures_{}_1.txt'.format(example_code), delimiter=',')

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

class Map_Elite:
    all = []
    
    def __init__(self, function, alphas =[0]):
        # function must acept a Level as input
        # fuction must have Image the real interval [0,1]
        # The Filtter will capting the optimal value (min)
        self.__N = len(alphas)
        self.__f = function
        self.__alphas = alphas
        self.__Opt_Values = np.ones(self.__N)
        self.__Optimal_strings = [""]*self.__N
            
    def Quest(self, Level):
        y = self.__f(Level)
        for i in range(self.__N):
            if self.__Optimal_strings[i]=="" or np.abs(self.__alphas[i]-y) < np.abs(self.__alphas[i]-self.__Opt_Values[i]):
                self.__Opt_Values[i] = y
                self.__Optimal_strings[i] = Level
    
    def Display_Range(self, a, b):
        if (a>b) or (b>self.__N) or (a*b<0):
            print("Display_Range Invalid index")
            return
        for i in range(a,b+1):
            Display_Level(self.__Optimal_strings[i])
        return
    
    def Report(self, Type = "Values"):
        if Type == "Values":
            return self.__Opt_Values
        if Type == "Strings":
            return self.__Optimal_strings
        else:
            return
    
    def __repr__(self):
        return [self.__Optimal_strings,self.__Opt_Values]
    
    def __len__(self):
        return self.__N



# Variety Dominess
def Landings_Score(Level):
            
    Count=0
    Max = 0
    for Key in list(Level):
        value = list(df.loc[df["Key"] == Key,"Landings" ])[0]
        Count += value
        if value>Max:
            Max = value
        
    return Count/(Max * len(Level))


# map-elites

Map_Landings_Score = Map_Elite(Landings_Score,alphas=[0,0.5,1])
Map_Performance= Map_Elite(pe.Performance,alphas=[0,0.1,0.5])

G = gn.Grammar(level_str)
Level = G.N_Level_Generator(Level_Len).__repr__()
while not(ws.Jumping_Fiasible_Word(Level)):
    Level = G.N_Level_Generator(Level_Len).__repr__()
Map_Landings_Score.Quest(Level)
Map_Performance.Quest(Level)

for j in range(1):
    print("Start",j)
    print("XX", "t", "Time", "--", "Performance", "--","--","--","--" "Landings_Score", "--","--","--","--", "Fiasible_Word %")
    base_time = time.time()
    t=0
    T=0
    for i in range(10000):
        pre_Level = G.N_Level_Generator(Level_Len).__repr__()
        T+=1
        while not(ws.Jumping_Fiasible_Word(pre_Level)):
            T+=1
            pre_Level = G.N_Level_Generator(Level_Len).__repr__()
        
        t+=1
        Map_Landings_Score.Quest(pre_Level)
        Map_Performance.Quest(pre_Level)
        
            
        if i%100 == 0:
            delta_time = time.time() - base_time
            print(i, "t", "%.2f" % (delta_time/60), "--", Map_Performance.Report(), "--","--", Map_Landings_Score.Report(), "--", "%.2f" % (t/T*100),"%")
        
    Map_Performance.Display_Range(0,2)
    Map_Landings_Score.Display_Range(0,2)

