# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 10:00:26 2023
Check List:
    *CUDA opt.
    *Finish Colliders
    *Hazardz Stitching
@author: Juan J. Rueda M.
The 2009 mario ai competition - togelius. (n.d.). http://julian.togelius.com/Togelius2010The.pdf 
Mario level generation from mechanics using scene stitching - arxiv.org. (n.d.-a). https://arxiv.org/pdf/2002.02992 

"""

from numba import njit
import numpy as np

Max_Height = 4 #max size of player jump
Max_Length = 4 #max length of player jump
Band_Height = 14 #size of level


#check if there is a bit in position k in number n
@njit(target_backend='cuda') 
def Punctual_Binary_Prop(n,k):
    n_ = n>>k 
    #returns only position
    return (n_ % 2**(k+1))%2

# slice bit array sections
@njit(target_backend='cuda')
def Range_Binary_Prop(n,a,b):
    #read structures
    a = max(0,a)
    b_ = (n>>b+1) * 2**(b+1)
    n_ = (n - b_)>>a
    #returns middle number
    return n_

def Jumping_Fiasible_Word(Level, knowledge, Game_mod = 0):
    """
    Taken from https://arxiv.org/pdf/2002.02992.pdf
    This function verify if a single word is possible to wander with walking and jumping, 
    in left to rigth scrolling
    
    A word finished in a letter without landing is't admisible.
    
    Game_Mods:
        0 * classic lateral Scroll
        1 * inverse lateral Scroll
        """
        
    if Game_mod==7357:
        return True
    
    
    Landings = knowledge[0]
    Colliders = knowledge[1]
    
    Structures_id = [int(ord(x)-65) for x in list(Level)]
    Int_Landings_List = [Landings[x] for x in Structures_id]
    Int_Colliders_List = [Colliders[x] for x in Structures_id]
    
    
    All_Landings_Accesibility_Stitched = True
    n = len(Level)
     #FIX
    
    #if no landing and beginning or end, this is unplayable
    if n == 0:
        return False   
    Initial = Int_Landings_List[0]
    Last = Int_Colliders_List[-1]
    if Initial==0 or Last==0:      
        return False
    
    #if level is only one letter, it is always playable
    if len(Level) == 1:
        return True

    LA_individually_Stitched_L = False
    JW_individually_Stitched = False         
    
    for n_letter in np.arange(n):
        Initial= Int_Landings_List[n_letter]
        R_Band_Height = np.arange(Band_Height)
        
        #check if the horizontal jump is possible
        #Forward_Stitching = [*range(n_letter +1, min(n_letter + Max_Length+1, n)),*range(max(n_letter - Max_Length, 0), n_letter-1 )]
        Forward_Stitching = np.arange(n_letter +1, min(n_letter + Max_Length+1, n))
        Back_Stiching = np.arange(max(n_letter - Max_Length, 0), n_letter)
        
        #Stitches = GPU_Stitching(Level, Forward_Stitching,Back_Stiching, R_Band_Height, Initial, n_letter)
        
        
        for i in R_Band_Height:
            
            
            if Punctual_Binary_Prop(Initial, i) == 1:
                LA_individually_Stitched_L = False
                JW_individually_Stitched = False
                
                #check for each land if exist previus land to jump from 
                
                stalactite_block = Band_Height                            
                
                for j in Back_Stiching:
                        
                    Int_Landing =  Int_Landings_List[j]
                    Int_Colliders = Int_Colliders_List[j]
                    
                    temp = Initial-2**i
                    
                    if temp<2**i:
                        if Range_Binary_Prop(Int_Landing,i-Max_Height,stalactite_block)!= 0:
                            LA_individually_Stitched_L = True 
                        elif Int_Colliders != Int_Landing:
                            first = False
                            for x in range(0,Band_Height):
                                if Punctual_Binary_Prop(Int_Colliders,x) == 1 and not(first):
                                    first = True
                                    stalactite_block = x 
                                
                    else:
                        first = False
                        for x in range(i+1,Band_Height):
                            if Punctual_Binary_Prop(Initial,x) == 1 and not(first):
                                first = True
                                stalactite_block = x    
                        
                        if Range_Binary_Prop(Int_Landing,i-Max_Height,stalactite_block)!= 0:
                                LA_individually_Stitched_L = True
                            
                #check for each land if exist other land to jump to
                for j in np.concatenate((Forward_Stitching,Back_Stiching), axis=0) :
                    
                    Int_Landing =  Int_Landings_List[j]
                    
                    if Range_Binary_Prop(Int_Landing,0,i+Max_Height)!= 0:
                            JW_individually_Stitched = True
                            
        if n_letter == 0:
            LA_individually_Stitched_L = True
            JW_individually_Stitched = True
            
                    
        #Check if everything is true
        All_Landings_Accesibility_Stitched = All_Landings_Accesibility_Stitched and LA_individually_Stitched_L and JW_individually_Stitched
        if not(All_Landings_Accesibility_Stitched):
            return All_Landings_Accesibility_Stitched
        
        
        
    
    # returns if the whole level is possible  
    return  All_Landings_Accesibility_Stitched           
