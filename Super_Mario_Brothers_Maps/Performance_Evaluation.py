# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:13:44 2023

@author: PC
"""
import pandas as pd
import Grammar_Notation as gn

def Performance(level):
    return list(level).count("E")*10/len(level)

Levels_Performance = pd.DataFrame(columns=["Level_String","Added_Wordlist", "Performance"])

def Add_Level_To_Performance(level, WordsList = [], Levels_Performance = Levels_Performance):
    G_ = gn.Grammar(level, WordsList)
    
    level_G_ = G_.N_Level_Generator(100).__repr__()
    new_row = [level, WordsList, Performance(level_G_)]
    Levels_Performance.loc[len(Levels_Performance)] = new_row
    
