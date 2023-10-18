# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 10:00:26 2023

@author: PC

"""
import sys
sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB")
import pandas as pd



def Punctual_Binary_Prop(n,k):
    n_ = n>>k 
    return (n_ % 2**(k+1))%2

def Range_Binary_Prop(n,a,b):
    a = max(0,a)
    b_ = (n>>b+1) * 2**(b+1)
    n_ = (n - b_)>>a
    return n_

df = pd.read_csv(r'C:\Users\PC\Documents\GitHub\EMPIS LAB\Structures_mix_1.txt', delimiter=',')


def Jumping_Fiasible_Word(word , Max_Height = 4, Max_Length = 4, Band_Height = 14, Game_mod=0):
    
    """This function verify if a single word is possible to wander with walking and jumping, 
    in left to rigth scrolling
    
    * Max_Height is the max Height of a single jump, 
    * Max_Length is the max Length of a single jump, 
    * Band_Height is the Height of the Display
    
    A word finished in a letter without landing is't admisible.
    
    Game_Mods:
        0 * classic lateral Scroll
        1 * Exploration
        2 * inverse lateral Scroll
        """
    
    Level= list(word) 
    Jump_Walk_Stitched = True
    All_Landings_Accesibility_Stitched = True
    
    Initial_0 = df["Landings"].loc[df["Key"] == Level[0]].iloc[0] #FIX
    
    if Initial_0==0:
        return False
    
    for n_letter in range(len(Level)):
        Initial = df["Landings"].loc[df["Key"] == Level[n_letter]].iloc[0] #FIX
        JW_individually_Stitched = False
        LA_individually_Stitched_R = False
        LA_individually_Stitched_L = False
        Path_Blocked = False
        
        #Horizontal_Range = [*range(n_letter +1, min(n_letter + Max_Length+1, len(Level))),*range(max(n_letter - Max_Length, 0), n_letter-1 )]
        Horizontal_Range = range(n_letter +1, min(n_letter + Max_Length+1, len(Level)))
        Back_Stiching = range(max(n_letter - Max_Length, 0), n_letter)
        
        for i in range(Band_Height):
            if Punctual_Binary_Prop(Initial, i) == 1:
                """No estoy contento con esto"""
                LA_individually_Stitched_R = False
                LA_individually_Stitched_L = False
                JW_individually_Stitched = False
                for j in Horizontal_Range:
                    Int_Landing = df["Landings"].loc[df["Key"] == Level[j]].iloc[0] #FIX
                    Int_Collider = df["Colliders"].loc[df["Key"] == Level[j]].iloc[0] #FIX
                    if not(Path_Blocked):
                            
                        if Range_Binary_Prop(Int_Landing,0,i+Max_Height)!= 0:
                            JW_individually_Stitched = True
                                        
                        if not(JW_individually_Stitched) and (Int_Landing != Int_Collider):
                            #Back Stitching
                            """for b in Back_Stiching:
                                    Int_Landing_b = df["Landings"].loc[df["Key"] == Level[b]].iloc[0] #FIX
                                    
                                    for c in range(Band_Height):
                                        if Punctual_Binary_Prop(Int_Landing, c) == 1:
                                            for k in range(Max_Height):
                                                if c+k <= Band_Height and Punctual_Binary_Prop(Int_Landing_b, c + k) == 1:
                                                    JW_individually_Stitched = True
                                                if c-k >= 0 and Punctual_Binary_Prop(Int_Landing_b, c - k) == 1:
                                                    JW_individually_Stitched = True"""
                            Path_Blocked = True
                            JW_individually_Stitched = True #Temp
                        
                        if j ==  min(n_letter + Max_Length+1, len(Level))-1:
                            Path_Blocked = True
                                
                    if Range_Binary_Prop(Int_Landing,i-Max_Height,i)!= 0:
                            LA_individually_Stitched_R = True
                                                        
                for j in Back_Stiching:
                        
                    Int_Landing = df["Landings"].loc[df["Key"] == Level[j]].iloc[0] #FIX
                        
                    if Range_Binary_Prop(Int_Landing,i-Max_Height,i)!= 0:
                            LA_individually_Stitched_L = True
                            
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
                    
        All_Landings_Accesibility_Stitched = All_Landings_Accesibility_Stitched and LA_individually_Stitched_L and LA_individually_Stitched_R
        Jump_Walk_Stitched = Jump_Walk_Stitched and JW_individually_Stitched
        if not(Jump_Walk_Stitched and All_Landings_Accesibility_Stitched):
            return Jump_Walk_Stitched and All_Landings_Accesibility_Stitched
        
        
        
        Jump_Walk_Stitched = Jump_Walk_Stitched and JW_individually_Stitched
        
    return  Jump_Walk_Stitched and All_Landings_Accesibility_Stitched