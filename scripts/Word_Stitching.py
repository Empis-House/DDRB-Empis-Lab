# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 10:00:26 2023
Check List:
    *CUDA opt.
    *Finish Colliders
    *Hazardz Stitching
@author: Juan J. Rueda M.

"""
import sys
# sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps")
import pandas as pd
from numba import njit
import numpy as np

Max_Height = 4 #max size of player jump
Max_Length = 4 #max length of player jump
Band_Height = 14 #size of level

#read structures
df = pd.read_csv(r'../Super_Mario_Brothers_Maps/structures/Structures_mix_1.txt'.format("mix",1))

#pass info to numpy
Columns = df.columns.to_numpy()
dfnp = np.transpose(df.to_numpy())
# Estructures_Only = dfnp[0,2]
knowledge = dfnp[3:]

Landings = knowledge[0]
Colliders = knowledge[1]

#check if there is a bit in position k in number n
@njit(target_backend='cuda') 
def Punctual_Binary_Prop(n,k):
    n_ = n>>k 
    #returns only position
    return (n_ % 2**(k+1))%2

# slice bit array sections
@njit(target_backend='cuda')
def Range_Binary_Prop(n,a,b):
    a = max(0,a)
    b_ = (n>>b+1) * 2**(b+1)
    n_ = (n - b_)>>a
    #returns middle number
    return n_

def Jumping_Fiasible_Word(Level,Game_mod = 0):
    """
    Taken from https://arxiv.org/pdf/2002.02992.pdf
    This function verify if a single word is possible to wander with walking and jumping, 
    in left to rigth scrolling
    
    A word finished in a letter without landing is't admisible.
    
    Game_Mods:
        0 * classic lateral Scroll
        1 * Exploration
        2 * inverse lateral Scroll
        """
    
    Structures_id = [int(ord(x)-65) for x in list(Level)]
    Int_Landings_List = [Landings[x] for x in Structures_id]
    Int_Colliders_List = [Colliders[x] for x in Structures_id]
    
    Jump_Walk_Stitched = True
    All_Landings_Accesibility_Stitched = True
    n = len(Level)
    Initial = Int_Landings_List[0] #FIX
    
    #if no landing and beginning or end, this is unplayable
    if len(Level) == 0 or Initial==0:
        return False   
    #if level is only one letter, it is always playable
    if len(Level) == 1:
        return True                    
    for n_letter in np.arange(n):
        Initial= Int_Landings_List[n_letter]
        R_Band_Height = np.arange(Band_Height)
        
        #check if the horizontal jump is possible
        #Horizontal_Range = [*range(n_letter +1, min(n_letter + Max_Length+1, n)),*range(max(n_letter - Max_Length, 0), n_letter-1 )]
        Horizontal_Range = np.arange(n_letter +1, min(n_letter + Max_Length+1, n))
        Back_Stiching = np.arange(max(n_letter - Max_Length, 0), n_letter)
        
        #Stitches = GPU_Stitching(Level, Horizontal_Range,Back_Stiching, R_Band_Height, Initial, n_letter)
        
        Path_Blocked = False
        #check forward for landings
        for i in R_Band_Height:
            if Punctual_Binary_Prop(Initial, i) == 1:
                LA_individually_Stitched_R = False
                LA_individually_Stitched_L = False
                JW_individually_Stitched = False
                for j in Horizontal_Range:
                    Int_Landing = Int_Landings_List[j]
                    Int_Collider = Int_Colliders_List[j]
                    
                    #if there's free pass
                    if not(Path_Blocked):
                        if Range_Binary_Prop(Int_Landing,0,i+Max_Height)!= 0:
                            JW_individually_Stitched = True
                                        
                        if not(JW_individually_Stitched) and (Int_Landing != Int_Collider):
                            #Back Stitching
                            
                            Path_Blocked = True
                            JW_individually_Stitched = True #Temp
                        
                        if j ==  min(n_letter + Max_Length+1, n)-1:
                            Path_Blocked = True
                                
                    if Range_Binary_Prop(Int_Landing,i-Max_Height,i)!= 0:
                            LA_individually_Stitched_R = True
                
                #check backwards for returning path                                   
                for j in Back_Stiching:
                        
                    Int_Landing =  Int_Landings_List[j]
                        
                    if Range_Binary_Prop(Int_Landing,i-Max_Height,i)!= 0:
                            LA_individually_Stitched_L = True 
        
        #setting up specific cases   
        if Initial == 0:
            JW_individually_Stitched = True
            LA_individually_Stitched_R = True
            LA_individually_Stitched_L = True
            
        if n_letter == len(Level)-1:
            JW_individually_Stitched = True
            LA_individually_Stitched_R = True
            
        if n_letter == 0:
            LA_individually_Stitched_L = True
            
        if Game_mod == 0:
            LA_individually_Stitched_R = True
        elif Game_mod == 2:
            LA_individually_Stitched_L = True
                    
        #Check if everything is true
        All_Landings_Accesibility_Stitched = All_Landings_Accesibility_Stitched and LA_individually_Stitched_L and LA_individually_Stitched_R
        Jump_Walk_Stitched = Jump_Walk_Stitched and JW_individually_Stitched
        if not(Jump_Walk_Stitched and All_Landings_Accesibility_Stitched):
            return Jump_Walk_Stitched and All_Landings_Accesibility_Stitched
        
        
        
        Jump_Walk_Stitched = Jump_Walk_Stitched and JW_individually_Stitched
        
    
    # returns if the whole level is possible  
    return  Jump_Walk_Stitched and All_Landings_Accesibility_Stitched             
