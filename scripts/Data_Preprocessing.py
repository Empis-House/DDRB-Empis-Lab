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

import pandas as pd
# import Grammar_Notation as gn

############ Limitation: ASCII is a 7-bit character set containing 128 characters.
#### Solution: The grammars could be closed-managed, i.e. We can difine a grammar with only 128 maximum structures,
#### but we can create as many as we want. Also if we guarantti that grammars match at least some structures, so that 
#### form a full-conected network, which could transitaded between grammars with >128 structures considered. I don't konw the bound yet.


#Function to join elements of a row into one string
def Structure(row):
    return "".join(row[6:]) 

#Function to check landings in a row and remove anything else
def Landing_Search(row,lands_ID = {"X", "S", "<", ">", "[", "]","B","b","Q","?"}): #Each mechanic stitching must have a Fi-2pop evaluation
    boolean_lands =""
    for i in range(1,14): #this range is dependant on level size
        #if character is a landing, it stays
        if (row["{}".format(i)] in lands_ID) and not(row["{}".format(i-1)] in lands_ID): # 
            boolean_lands += "1"
        #if it's anything else, it is removed to nothing    
        else: boolean_lands += "0"
        
    return int(boolean_lands,2)

#function to check "colliders" or tiles that you can't pass through
def Colliders_Search (row,collider_ID = {"X", "S", "<", ">", "[", "]","B","b","Q","?"}):
    boolean_colliders = "".join(row.apply(lambda x: "1" if x in collider_ID else "0"))
    return int(boolean_colliders,2)

def Rewards_Search (row,collider_ID = {"o","?","Q"}):
    boolean_colliders = "".join(row.apply(lambda x: "1" if x in collider_ID else "0"))
    return int(boolean_colliders,2)

def Hazard_Search (row,collider_ID = {"E","B"}):
    boolean_colliders = "".join(row.apply(lambda x: "1" if x in collider_ID else "0"))
    return int(boolean_colliders,2)

def Save_Mix_Structures(Ex_Levels_List):
        
    #Base Dataframes to store the level's structure
    ## Keys_Structures: barebones form of the level. Only Landings and Colliders
    ## Complete_Level: The full level completely processed
    ### Key: symbols that represents that particular row
    ### Structures: the string of the row
    ### Landings: Binary code for where the landings are
    ### Colliders: Binary code for where are colliders are 
    Keys_Structures = pd.DataFrame(columns=["token", "Structures", "Landings", "Colliders","Reward","Hazard"])
    Complete_Level = pd.DataFrame(columns=["Structures","token","Landings", "Colliders","Reward","Hazard"] + ["{}".format(i) for i in range(13)])
    Temporal = Complete_Level
    
    # Get the structure for each row
    for example_code in Ex_Levels_List:
        Temporal = pd.DataFrame(columns=["Structures","token","Landings", "Colliders"] + ["{}".format(i) for i in range(13)])
        with open(r"Super_Mario_Brothers_Maps/Processed/mario-{}.txt".format(example_code)) as infile:
            i = 0
            for line in infile: 
                Temporal["{}".format(i)] = list(line.split()[0])
                i+=1
                
        Complete_Level = pd.concat([Complete_Level,Temporal], axis=0)
        
    print(Complete_Level)
    Complete_Level["Structures"] = Complete_Level.apply(Structure, axis=1)
    Complete_Level["Landings"] = Complete_Level.apply(Landing_Search, axis=1)
    Complete_Level["Colliders"] = Complete_Level.apply(Colliders_Search, axis=1)
    Complete_Level["Reward"] = Complete_Level.apply(Rewards_Search, axis=1)
    Complete_Level["Hazard"] = Complete_Level.apply(Hazard_Search, axis=1)
    
    #remove duplicate structures
    Keys_Structures[["Structures","Landings","Colliders","Reward","Hazard"]] = Complete_Level[["Structures","Landings","Colliders","Reward","Hazard"]].drop_duplicates()
    Keys_Structures = Keys_Structures.reset_index()
    
    #Set the keys from ASCII
    for index in Keys_Structures.index:
        Keys_Structures.loc[index,"token"] = index
    
    Keys_Structures.to_csv(r'Super_Mario_Brothers_Maps/structures/Structures_{}.txt'.format(Ex_Levels_List), index=False)

    
Save_Mix_Structures(['1-1','1-2','1-3','2-1','3-1','3-3','4-1','4-2','5-1','5-3','6-1','6-2','6-3','7-1','8-1'])