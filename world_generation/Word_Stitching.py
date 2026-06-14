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


#check if there is a 1 in position k of the number n
# 4 = 00100000000000 so p(n=4,k=2) = 1 and p(n=4,k!=2) = 0
#@njit(target_backend='cuda') 
def Punctual_Binary_Prop(n,k):
    n_ = n>>k 
    #returns only position
    return (n_ % 2**(k+1))%2

# slice bit array sections
#@njit(target_backend='cuda')
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

    Structures_id = [x for x in list(Level)]
    Int_Landings_List = [Landings[x] for x in Level]
    Int_Colliders_List = [Colliders[x] for x in Level]
    
    def First_One_Above(mask, start):
        for height in range(start, Band_Height):
            if Punctual_Binary_Prop(mask, height) == 1:
                return height
        return None

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
            if Punctual_Binary_Prop(Initial, i) != 1:
                continue

            LA_individually_Stitched_L = False
            JW_individually_Stitched = False

            for j in Back_Stiching:
                Int_Landing = Int_Landings_List[j]
                Int_Colliders = Int_Colliders_List[j]
                if Int_Landing == 0:
                    continue

                top_block = Band_Height
                next_landing_above = First_One_Above(Initial, i + 1)
                if next_landing_above is not None:
                    top_block = min(top_block, next_landing_above)

                next_collider_above = First_One_Above(Int_Colliders, i + 1)
                if next_collider_above is not None:
                    top_block = min(top_block, next_collider_above)

                min_height = max(0, i - Max_Height)
                max_height = min(top_block - 1, i + Max_Height)

                if max_height >= min_height and Range_Binary_Prop(Int_Landing, min_height, max_height) != 0:
                    LA_individually_Stitched_L = True
                    break

            for j in np.concatenate((Forward_Stitching, Back_Stiching), axis=0):
                Int_Landing = Int_Landings_List[j]
                if Range_Binary_Prop(Int_Landing, 0, i + Max_Height) != 0:
                    JW_individually_Stitched = True
                    break

            if n_letter == 0:
                continue

            if not LA_individually_Stitched_L or not JW_individually_Stitched:
                return False

        if n_letter == 0:
            LA_individually_Stitched_L = True
            JW_individually_Stitched = True

        All_Landings_Accesibility_Stitched = All_Landings_Accesibility_Stitched and LA_individually_Stitched_L and JW_individually_Stitched
        if not All_Landings_Accesibility_Stitched:
            return False
        
        
        
    
    # returns if the whole level is possible  
    return  All_Landings_Accesibility_Stitched           
