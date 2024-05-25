# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:31:17 2023

@author: PC
"""

{
    "tiles" : {
        "X" : ["solid","ground"],
        "S" : ["solid","breakable"],
        "-" : ["passable","empty"],
        "?" : ["solid","question block", "full question block"],
        "Q" : ["solid","question block", "empty question block"],
        "E" : ["enemy","damaging","hazard","moving"],
        "<" : ["solid","top-left pipe","pipe"],
        ">" : ["solid","top-right pipe","pipe"],
        "[" : ["solid","left pipe","pipe"],
        "]" : ["solid","right pipe","pipe"],
        "o" : ["coin","collectable","passable"],
        "B" : ["Cannon top","cannon","solid","hazard"],
        "b" : ["Cannon bottom","cannon","solid"]
    }
}

import sys
sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps")
import pandas as pd
from numba import njit


from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import numpy as np
import scripts.Grammar_Notation as gn
import time
from timeit import default_timer as timer 

Max_Height = 4 
Max_Length = 4
Band_Height = 14
@njit(target_backend='cuda') 
def Punctual_Binary_Prop(n,k):
    n_ = n>>k 
    return (n_ % 2**(k+1))%2
@njit(target_backend='cuda')
def Range_Binary_Prop(n,a,b):
    a = max(0,a)
    b_ = (n>>b+1) * 2**(b+1)
    n_ = (n - b_)>>a
    return n_


df = pd.read_csv(r'C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps\Structures_{}_{}.txt'.format("mix",1))

Columns = df.columns.to_numpy()
dfnp = np.transpose(df.to_numpy())
# normal function to run on cpu
Estructures_Omly = dfnp[0,2]
Knowlendg = dfnp[3:]

Landings = Knowlendg[0]
Colliders = Knowlendg[1]
print(np.size(Landings))


del(df)

def func(Level): 
    Jump_Walk_Stitched = True
    All_Landings_Accesibility_Stitched = True

    Initial = Landings[ord(Level[0])-65] #FIX
    if Initial ==0:
        return NotImplementedError() 
                              
    for n_letter in np.arange(len(Level)):
        Initial= Landings[ord(Level[n_letter])-65]
        R_Band_Height = np.arange(Band_Height)
        
        #Horizontal_Range = [*range(n_letter +1, min(n_letter + Max_Length+1, len(Level))),*range(max(n_letter - Max_Length, 0), n_letter-1 )]
        Horizontal_Range = np.arange(n_letter +1, min(n_letter + Max_Length+1, len(Level)))
        Back_Stiching = np.arange(max(n_letter - Max_Length, 0), n_letter)
        
        #Stitches = GPU_Stitching(Level, Horizontal_Range,Back_Stiching, R_Band_Height, Initial, n_letter)
        
        Path_Blocked = False
        for i in R_Band_Height:
            if Punctual_Binary_Prop(Initial, i) == 1:
                LA_individually_Stitched_R = False
                LA_individually_Stitched_L = False
                JW_individually_Stitched = False
                for j in np.array(Horizontal_Range):
                    Int_Landing = Landings[ord(Level[j])-65]
                    Int_Collider = Colliders[ord(Level[j])-65]
                    if not(Path_Blocked):
                        if Range_Binary_Prop(Int_Landing,0,i+Max_Height)!= 0:
                            JW_individually_Stitched = True
                                        
                        if not(JW_individually_Stitched) and (Int_Landing != Int_Collider):
                            #Back Stitching
                            
                            Path_Blocked = True
                            JW_individually_Stitched = True #Temp
                        
                        if j ==  min(n_letter + Max_Length+1, len(Level))-1:
                            Path_Blocked = True
                                
                    if Range_Binary_Prop(Int_Landing,i-Max_Height,i)!= 0:
                            LA_individually_Stitched_R = True
                                                        
                for j in np.array(Back_Stiching):
                        
                    Int_Landing =  Landings[ord(Level[j])-65]
                        
                    if Range_Binary_Prop(Int_Landing,i-Max_Height,i)!= 0:
                            LA_individually_Stitched_L = True 
                            
        Jump_Walk_Stitched = True
        All_Landings_Accesibility_Stitched = True
        
 
# function optimized to run on gpu 

#@njit(target_backend='cuda')                         
def func2(Level, Int_Landings_List, Int_Colliders_List):
    Jump_Walk_Stitched = True
    All_Landings_Accesibility_Stitched = True
    n = len(Level)
    Initial = Int_Landings_List[0] #FIX
    
    if Initial == 0:
        return                              
    for n_letter in np.arange(n):
        Initial= Int_Landings_List[n_letter]
        R_Band_Height = np.arange(Band_Height)
        
        #Horizontal_Range = [*range(n_letter +1, min(n_letter + Max_Length+1, n)),*range(max(n_letter - Max_Length, 0), n_letter-1 )]
        Horizontal_Range = np.arange(n_letter +1, min(n_letter + Max_Length+1, n))
        Back_Stiching = np.arange(max(n_letter - Max_Length, 0), n_letter)
        
        #Stitches = GPU_Stitching(Level, Horizontal_Range,Back_Stiching, R_Band_Height, Initial, n_letter)
        
        Path_Blocked = False
        for i in R_Band_Height:
            if Punctual_Binary_Prop(Initial, i) == 1:
                LA_individually_Stitched_R = False
                LA_individually_Stitched_L = False
                JW_individually_Stitched = False
                for j in Horizontal_Range:
                    Int_Landing = Int_Landings_List[j]
                    Int_Collider = Int_Colliders_List[j]
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
                                                        
                for j in Back_Stiching:
                        
                    Int_Landing =  Int_Landings_List[j]
                        
                    if Range_Binary_Prop(Int_Landing,i-Max_Height,i)!= 0:
                            LA_individually_Stitched_L = True 
                            
        Jump_Walk_Stitched = True
        All_Landings_Accesibility_Stitched = True
    
G = gn.Grammar("AABABBCDAEAEABABAABAAWSXMYZAAFFAAETEAAETEAASABASAAEETTEAETTEAAHIAAKLAAESEMEZE[[AAWAA")       
Level_Len = 20

n = 100
base_time = time.time()

    
start = timer()
m=0
l=0
for i in range(n):              
    word = G.N_Level_Generator(Level_Len).__repr__()
    Level= np.array(list(word)) 
     
    
    start = timer()
    func(Level)
    m += timer()-start
    
    start = timer()             
    func2(Level, Int_Landings, Int_Colliders)
    l += timer()-start
    
print("without GPU:", m)
print("with GPU:", l)
    
    
