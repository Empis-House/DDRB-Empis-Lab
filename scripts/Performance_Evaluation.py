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
    if len(level)==0:
        return NotImplementedError()
    Count=0
    array = ["E","Q","T","d","i","k","v"]
    for x in array:
        Count = Count + list(level).count(x)
    return Count/len(level)

Levels_Performance = pd.DataFrame(columns=["Level_String","Added_Wordlist", "Performance"])

def Add_Level_To_Performance(level, WordsList = [], Levels_Performance = Levels_Performance):
    G_ = gn.Grammar(level, WordsList)
    
    level_G_ = G_.N_Level_Generator(100).__repr__()
    new_row = [level, WordsList, Performance(level_G_)]
    Levels_Performance.loc[len(Levels_Performance)] = new_row
    
