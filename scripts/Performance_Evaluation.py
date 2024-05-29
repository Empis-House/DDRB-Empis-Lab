# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:13:44 2023
Check List:
    *Planning
@author: Juan J. Rueda M.
"""
import pandas as pd
import scripts.Grammar_Notation as gn

def Performance(level,df):
    Count=0
    for Key in level:
        try:
            value = list(df.loc[df["token"].astype(int) == Key,"Landings"])[0]
            if value == 0:
                Count += 5/3 # The way generation is now configured guarantees a greater ratio of 3 supports for every 5 landing pieces.
        except Exception as e:
            print(Key, type(Key),"      ", e)
    return Count/len(level)

Levels_Performance = pd.DataFrame(columns=["Level_String","Added_Wordlist", "Performance"])

def Add_Level_To_Performance(level, WordsList = [], Levels_Performance = Levels_Performance):
    G_ = gn.Grammar(level, WordsList)
    
    level_G_ = G_.N_Level_Generator(100).__repr__()
    new_row = [level, WordsList, Performance(level_G_)]
    Levels_Performance.loc[len(Levels_Performance)] = new_row


    
