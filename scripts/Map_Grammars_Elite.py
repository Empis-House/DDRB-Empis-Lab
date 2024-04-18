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

import Grammar_Notation as gn
import Word_Stitching as ws
import Performance_Evaluation as pe
import pandas as pd
import time
import numpy as np

seed=None
example_code = ["1-1","4-1"]
Level_Len = 80
module = 80
Game_Mode = 0

#read structures
df = pd.read_csv(r'../Super_Mario_Brothers_Maps/structures/Structures_{}.txt'.format(example_code))

#pass info to numpy
Columns = df.columns.to_numpy()
dfnp = np.transpose(df.to_numpy())
Knowledge = dfnp[3:]



#display and save final level if name is included
def Display_Level(Level, level_name=None, df = df):
    
    print("\n",pe.Performance(Level))
    print(Landings_Score(Level))
    print(ws.Jumping_Fiasible_Word(Level,knowledge = Knowledge))
            
    Display = pd.DataFrame(columns= ["{}".format(i) for i in range(14)])
    for Key in list(Level):
        new_row = list(list(df.loc[df["Key"] == Key,"Structures" ])[0])
        Display.loc[len(Display)] = new_row
        
        
    Display = Display.transpose()
    if level_name != None:
        Display.to_csv(r'../Super_Mario_Brothers_Maps/final_levels/{}.txt'.format(level_name), index=False,header=False,sep=",")
    
    print(Display)

#CReate Map Elite Class
#Good explanation on https://arxiv.org/abs/2205.05834
class Map_Elite:
    all = []
    
    def __init__(self, function, alphas =[0], Variety_Dominess = [], Grid_points =[]):
        # function must acept a Level as input
        # fuction must have Image the real interval [0,1]
        # The Filtter will capting the optimal value (min)
        self.__N = len(alphas)
        self.__f = function
        self.__alphas = alphas
        self.__Variety_Dominess = Variety_Dominess
        
        self.__Opt_Values = None
        self.__Optimal_strings = None
        
        self.__Grid_points = Grid_points
        
        if not(self.__Variety_Dominess):
            self.__Opt_Values = np.ones(self.__N)
            self.__Optimal_strings = [""]*self.__N
            
        elif len(Variety_Dominess) == len(self.__Grid_points):
            
            return
            
                
        else:
            print("The Variety Grid failse, the Map have set as Default")
            self.__Opt_Values = np.ones(self.__N)
            self.__Optimal_strings = [""]*self.__N
        
            
    def Quest(self, Level):
        y = self.__f(Level)
        
        if not(self.__Variety_Dominess):
            for i in range(self.__N):
                if self.__Optimal_strings[i]=="" or np.abs(self.__alphas[i]-y) < np.abs(self.__alphas[i]-self.__Opt_Values[i]):
                    self.__Opt_Values[i] = y
                    self.__Optimal_strings[i] = Level
            return
        
        
        """for i in range(len(Level_Vector)):
            p=1
            coordinate = None
            print(self.__Optimal_strings[i])
            print(Level_Vector)
            
            for j in range(len(self.__Grid_points[i])):
                if abs(Level_Vector[i]-self.__Grid_points[i][j])<p:
                    p = Level_Vector[i]-self.__Grid_points[i][j]"""
                    
            
            
        
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

def Structure(row):
  return "".join(row[2:])

def Key_substraction(x, df=df):
    "-------------X"
    "---------XXXXX"
    "--------------"
    "-------XXXXXXX"
    if len(df[df["Structures"] ==  x]["Key"])==0:
        return "?"
    else:
        return np.array(df[df["Structures"] ==  x]["Key"])[0]

def Extract_Level_String(examples_codes,df=df):
    for example_code in examples_codes:
        Temporal = pd.DataFrame(columns=["Structures","Key"] + ["{}".format(i) for i in range(13)])
        with open(r"../Super_Mario_Brothers_Maps/Processed/mario-{}.txt".format(example_code)) as infile:
            i = 0
            for line in infile: 
                Temporal["{}".format(i)] = list(line.split()[0])
                i+=1
        Level = pd.DataFrame(columns=["Structures","Key"] + ["{}".format(i) for i in range(13)])
        Level = pd.concat([Level,Temporal], axis=0)
             
    Level["Structures"] = Level.apply(Structure, axis=1)
    
    for i in range(len(Level["Structures"])):
        Level["Key"][i]= Key_substraction(Level["Structures"][i])
        
    return "".join(Level["Key"])



# Variety Dominess
def Landings_Score(Level):
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


# map-elites

Map_Landings_Score = Map_Elite(Landings_Score,alphas=[0,1])#, Variety_Dominess=[pe.Performance],Grid_points=[[0,0.1,0.25,0.5]])
Map_Performance= Map_Elite(pe.Performance,alphas=[0,0.1,1])

level_str = Extract_Level_String([example_code[0]]) +" "+ Extract_Level_String([example_code[1]])

G = gn.Grammar(level_str,knowledge = Knowledge)
pre_Level = G.N_Level_Generator(Level_Len,module=module,Gen_Game_mod=Game_Mode).__repr__()
while not(ws.Jumping_Fiasible_Word(pre_Level, knowledge = Knowledge, Game_mod=Game_Mode)):
    pre_Level = G.N_Level_Generator(Level_Len,module=module,Gen_Game_mod=Game_Mode).__repr__()
Map_Landings_Score.Quest(pre_Level)
Map_Performance.Quest(pre_Level)


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
        Map_Landings_Score.Quest(pre_Level)
        Map_Performance.Quest(pre_Level)
        
            
        if i%100 == 0:
            delta_time = time.time() - base_time
            print(i, "t", "%.2f" % (delta_time/60), "--", Map_Performance.Report(), "--","--", Map_Landings_Score.Report(), "--", "%.2f" % (t/T*100),"%")
    delta_time = time.time() - base_time
    print(i, "t", "%.2f" % (delta_time/60), "--", Map_Performance.Report(), "--","--", Map_Landings_Score.Report(), "--", "%.2f" % (t/T*100),"%")
    #Map_Performance.Display_Range(0,3)
    #Map_Landings_Score.Display_Range(0,1)
    

Display_Level(Map_Performance.Report(Type = "Strings")[0],level_name="Performance/{}_opt=0".format(example_code))
Display_Level(Map_Performance.Report(Type = "Strings")[1],level_name="Performance/{}_opt=0.1".format(example_code))  
Display_Level(Map_Performance.Report(Type = "Strings")[2],level_name="Performance/{}_opt=1".format(example_code))


Display_Level(Map_Landings_Score.Report(Type = "Strings")[0],level_name="Landings_Score/{}_opt=0".format(example_code))
Display_Level(Map_Landings_Score.Report(Type = "Strings")[1],level_name="Landings_Score/{}_opt=1".format(example_code))