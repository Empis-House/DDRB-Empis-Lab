# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:12:50 2023

@author: PC
"""

import pandas as pd
import random

class Word:
    
    def __init__(self, Sequence):
        self.__Sequence = Sequence
        
    def OpenKey(self):
        return self.__Sequence[0]
    
    def CloseKey(self):
        return self.__Sequence[-1]
    
    def __repr__(self):
        return f"{self.__Sequence}"
    
    def __add__(self, other):
        return Word(self.__Sequence + other.__repr__())
    
    def __len__(self):
        return len(self.__Sequence)
    
class Grammar:
    
    all = []
    
    def __init__(self, ExampleLevel, WordsList = []): 
        """Check List: 
        verify First type of CAG
        Time testing
        O review
        Reduced_Blacklist: not implemented case, Long word (redundance)  
        """
        
        self.__ExampleLevel = ExampleLevel 
        
        self.__WordsList = WordsList + list(ExampleLevel)   
        
        self.__Keys_Matrix = pd.DataFrame() #this Keys_Matrix resume all and only transition between keys 
        LettersSet = sorted(list(set(self.__ExampleLevel)))
        self.__Keys_Matrix["from"] = LettersSet
        
        last_letter = ''
        
        for letter in ExampleLevel:
            if not(letter in self.__Keys_Matrix.columns):
                self.__Keys_Matrix[letter] = [0]*len(LettersSet)
                
            if last_letter != '':
                x = self.__Keys_Matrix.index[self.__Keys_Matrix["from"] == last_letter].tolist()[0]
                y = self.__Keys_Matrix.columns.get_loc(letter)
                self.__Keys_Matrix.iloc[x,y] = 1
            
            last_letter = letter
            
        Grammar.all = Grammar.all + [self]
            
    def __repr__(self):
        return f"{self.__Keys_Matrix}\n"
    
    def Add_Word(self, word):
        self.__WordsList = self.__WordsList + [word]
        
    def Words_List(self):
        print(self.__WordsList)
    
    def Simple_words(self):
        
        All_Keys = list(self.__ExampleLevel)
        Simple_Words = set()
        
        for key in All_Keys:
            x = self.__Keys_Matrix.index[self.__Keys_Matrix["from"] == key].tolist()[0]                
                
            for k in  All_Keys:
                y = self.__Keys_Matrix.columns.get_loc(k)
                if self.__Keys_Matrix.iloc[x,y] == 1:
                    Simple_Words = Simple_Words.union({key + k})
                            
        return Simple_Words
    
    def Reduced_Blacklist(self, BlackList = ["FVBREDPAWSG","EEE"]): 
        Reducted_Blacklist = []
        for forbidden_word in BlackList:
            Filter = list(forbidden_word) #[E,E,E] or [F,V,B,R,E,D,P,A,W,S,G]
            In_Stack = []
            #Out_Stack = []s
            
            for word in self.__WordsList:
                if word in forbidden_word and len(word)>2: # ["BREDPAWSG"]
                    word_location = forbidden_word.find(word)
                    # If word = A then len(word)=1
                    # word_location + len(word)-1 = word_location 
                    # Then, word_location + 1 > word_location + len(word) - 1
                    # And useless if you what avoid a serten letter, delete it's node.
                    
                    # If word = AB then len(word)=2
                    # word_location + len(word) - 1 = word_location + 1
                    # Then, word_location + 1 = word_location + len(word)-1
                    # Is like do nothing in the Filter and useless too, to trotgh out AB is esear deled it's edge.
                    
                    Filter = Filter[0:word_location + 1] + Filter[word_location + len(word)-1:] 
                    # [F,V,"B,R,E,D,P,A,W,S,G"] To [F,V,B,G] which len is No. of 
                    
            
            for i in range(len(Filter)-1):
                if Filter[i] + Filter[i+1] in self.Simple_words():
                    In_Stack = In_Stack + [Filter[i] + Filter[i+1]]
                #else:
                    #Out_Stack = Out_Stack + [Filter[i] + Filter[i+1]]
                    
                    
            if len(In_Stack)+1 == len(Filter) :
                Reducted_Blacklist = Reducted_Blacklist + [forbidden_word]
                
        return Reducted_Blacklist

    def N_Level_Generator(self, N, Start = '', seed = None):

        random.seed(seed)
        
        if  Start == '':
            n = int(len(self.__WordsList)* random.random())
            New_Level = Word(self.__WordsList[n])
        elif Start in self.__WordsList:
            New_Level = Word(Start)
        else:
            print("invalid start")
            return
        
        while len(New_Level)<N:
            
            options = list()
            
            for word in self.__WordsList:
                word = Word(word)
                
                x = self.__Keys_Matrix.index[self.__Keys_Matrix["from"] == New_Level.CloseKey()].tolist()[0]
                y = self.__Keys_Matrix.columns.get_loc(word.OpenKey())
                
                if self.__Keys_Matrix.iloc[x,y] > 0:
                    options = options + [word]
            
            n = int(len(options)* random.random())
            New_Level = New_Level + options[n]
            
        return New_Level
    