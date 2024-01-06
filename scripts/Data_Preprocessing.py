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
# sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB")
import pandas as pd
# import Grammar_Notation as gn

############ Limitation: ASCII is a 7-bit character set containing 128 characters.
#### Solution: The grammars could be closed-managed, i.e. We can difine a grammar with only 128 maximum structures,
#### but we can create as many as we want. Also if we guarantti that grammars match at least some structures, so that 
#### form a full-conected network, which could transitaded between grammars with >128 structures considered. I don't konw the bound yet.

seed=1 #pseudo randomness
examples_codes = ["8-1","6-3"] #super mario levels where we take the structure
Level_Len = 500 #full size of the generated level

#Function to join elements of a row into one string
def Structure(row):
    return "".join(row[4:]) #this number is dependant of the level

#Function to check landings in a row and remove anything else
def Landing_Search(row,lands_ID = {"X", "S", "<", ">", "[", "]"}): #Each mechanic stitching must have a Fi-2pop evaluation
    boolean_lands =""
    for i in range(1,14): #this range is dependant on level size
        #if character is a landing, it stays
        if (row["{}".format(i)] in lands_ID) and not(row["{}".format(i-1)] in lands_ID): # 
            boolean_lands += "1"
        #if it's anything else, it is removed to nothing    
        else: boolean_lands += "0"
        
    return int(boolean_lands,2)

#function to check "colliders" or tiles that you can't pass through
def Colliders_Search (row,collider_ID = {"X", "S", "<", ">", "[", "]"}):
    boolean_colliders = "".join(row.apply(lambda x: "1" if x in collider_ID else "0"))
    return int(boolean_colliders,2)


#Base Dataframes to store the level's structure
## Keys_Structures: barebones form of the level. Only Landings and Colliders
## Complete_Level: The full level completely processed
### Key: symbols that represents that particular row
### Structures: the string of the row
### Landings: Binary code for where the landings are
### Colliders: Binary code for where are colliders are 
Keys_Structures = pd.DataFrame(columns=["Key", "Structures", "Landings", "Colliders"])
Complete_Level = pd.DataFrame(columns=["Structures","Key","Landings", "Colliders"] + ["{}".format(i) for i in range(13)])
Temporal = Complete_Level
Original_Level_Strings = pd.DataFrame(columns=["Level name", "Word"]) # No implemented

# Get the structure for each row
for example_code in examples_codes:
    Temporal = pd.DataFrame(columns=["Structures","Key","Landings", "Colliders"] + ["{}".format(i) for i in range(13)])
    with open(r"Super_Mario_Brothers_Maps/Processed/mario-{}.txt".format(example_code)) as infile:
        i = 0
        for line in infile: 
            Temporal["{}".format(i)] = list(line.split()[0])
            i+=1
            
    Complete_Level = pd.concat([Complete_Level,Temporal], axis=0)
    
  
Complete_Level["Structures"] = Complete_Level.apply(Structure, axis=1)
Complete_Level["Landings"] = Complete_Level.apply(Landing_Search, axis=1)
Complete_Level["Colliders"] = Complete_Level.apply(Colliders_Search, axis=1)
print(Complete_Level)

#remove duplicate structures
Keys_Structures[["Structures","Landings","Colliders"]] = Complete_Level[["Structures","Landings","Colliders"]].drop_duplicates()
Keys_Structures = Keys_Structures.reset_index()
Keys_Structures["Test"] = Keys_Structures.apply(lambda row: row["Colliders"] - row["Landings"], axis=1)

#Set the keys from ASCII
for index in Keys_Structures.index:
    Keys_Structures.loc[index,"Key"] = "{}".format(chr(65+index))

print(Keys_Structures)

Keys_Structures.to_csv(r'Super_Mario_Brothers_Maps/structures/Structures_{}_{}.txt'.format("mix",seed), index=False)

    
# """G = gn.Grammar("AABABBCDAEAEABABAABAAWSXMYZAAFFAAETEAAETEAASABASAAEETTEAETTEAAHIAAKLAAESEMEZE[[AAWAA")
# Level = G.N_Level_Generator(Level_Len,seed = seed).__repr__()
# Display = pd.DataFrame(columns= ["{}".format(i) for i in range(14)])
# for Key in list(Level):
#     new_row = list(list(Keys_Structures.loc[Keys_Structures["Key"] == Key,"Structures" ])[0])
#     Display.loc[len(Display)] = new_row
    
# Display = Display.transpose()
# Display.to_csv(r'C:\Users\PC\Documents\GitHub\EMPIS LAB\{}_{}.txt'.format("mix",seed), index=False,header=False)
# print(Display) 

# Display.to_csv(r'C:\Users\PC\Documents\GitHub\EMPIS LAB\{}_{}.txt'.format("mix",seed), index=False,header=False)
# """
