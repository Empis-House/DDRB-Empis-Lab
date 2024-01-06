# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:10:49 2023
Check List:
    *Extract Example Levels strings
    *Hazards and Rewards Search
    *Hazards and Rewards _Score implementation in Performance_Evaluation
    *Reward decoupling
@author: Juan J. Rueda M.
"""

import sys

#sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps")
sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB")
import pandas as pd
import Grammar_Notation as gn

############ Limitation: ASCII is a 7-bit character set containing 128 characters.
#### Solution: The grammars could be closed-managed, i.e. We can difine a grammar with only 128 maximum structures,
#### but we can create as many as we want. Also if we guarantti that grammars match at least some structures, so that 
#### form a full-conected network, which could transitaded between grammars with >128 structures considered. I don't konw the bound yet.

seed=1
examples_codes = ["8-1","6-3"]
Level_Len = 500

def Structure(row):
    
  return "".join(row[4:])

def Landing_Search(row,lands_ID = {"X", "S", "<", ">", "[", "]"}): #Each mecaniques stitching must have a Fi-2pop evaluation

    boolean_lands =""
    for i in range(1,14):
        if (row["{}".format(i)] in lands_ID) and not(row["{}".format(i-1)] in lands_ID):
            boolean_lands += "1"
            
        else: boolean_lands += "0"
        
    return int(boolean_lands,2)

def Colliders_Search (row,collider_ID = {"X", "S", "<", ">", "[", "]"}):
    boolean_colliders = "".join(row.apply(lambda x: "1" if x in collider_ID else "0"))
    return int(boolean_colliders,2)



Keys_Structures = pd.DataFrame(columns=["Key", "Structures", "Landings", "Colliders"])
Complite_Level = pd.DataFrame(columns=["Structures","Key","Landings", "Colliders"] + ["{}".format(i) for i in range(13)])
Temporal = Complite_Level
Original_Level_Strings = pd.DataFrame(columns=["Level name", "Word"]) # No implemented

for example_code in examples_codes:
    Temporal = pd.DataFrame(columns=["Structures","Key","Landings", "Colliders"] + ["{}".format(i) for i in range(13)])
    with open(r"C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps\Processed\mario-{}.txt".format(example_code)) as infile:
        i = 0
        for line in infile: 
            Temporal["{}".format(i)] = list(line.split()[0])
            i+=1
            
    Complite_Level = pd.concat([Complite_Level,Temporal], axis=0)
    
         
Complite_Level["Structures"] = Complite_Level.apply(Structure, axis=1)
Complite_Level["Landings"] = Complite_Level.apply(Landing_Search, axis=1)
Complite_Level["Colliders"] = Complite_Level.apply(Colliders_Search, axis=1)
print(Complite_Level)

Keys_Structures[["Structures","Landings","Colliders"]] = Complite_Level[["Structures","Landings","Colliders"]].drop_duplicates()
Keys_Structures = Keys_Structures.reset_index()
Keys_Structures["Test"] = Keys_Structures.apply(lambda row: row["Colliders"] - row["Landings"], axis=1)

for index in Keys_Structures.index:
    Keys_Structures.loc[index,"Key"] = "{}".format(chr(65+index))

print(Keys_Structures)

Keys_Structures.to_csv(r'C:\Users\PC\Documents\GitHub\EMPIS LAB\Structures_{}_{}.txt'.format("mix",seed), index=False)

    
"""G = gn.Grammar("AABABBCDAEAEABABAABAAWSXMYZAAFFAAETEAAETEAASABASAAEETTEAETTEAAHIAAKLAAESEMEZE[[AAWAA")
Level = G.N_Level_Generator(Level_Len,seed = seed).__repr__()
Display = pd.DataFrame(columns= ["{}".format(i) for i in range(14)])
for Key in list(Level):
    new_row = list(list(Keys_Structures.loc[Keys_Structures["Key"] == Key,"Structures" ])[0])
    Display.loc[len(Display)] = new_row
    
Display = Display.transpose()
Display.to_csv(r'C:\Users\PC\Documents\GitHub\EMPIS LAB\{}_{}.txt'.format("mix",seed), index=False,header=False)
print(Display) 

Display.to_csv(r'C:\Users\PC\Documents\GitHub\EMPIS LAB\{}_{}.txt'.format("mix",seed), index=False,header=False)"""
