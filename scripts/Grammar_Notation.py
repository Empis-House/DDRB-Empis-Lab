# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:12:50 2023
Check List: 
    *Perceptron (max Fiasible_Word %) Surrugate FI-2Pop Search
    *verify First type of CAG
    *Time testing
    *O review
    *Reduced_Blacklist: not implemented case, Long word (redundance)  
@author: Juan J. Rueda M.

References:
    Guzdial, M., Snodgrass, S., & Summerville, A. (2022). Procedural content generation via machine learning. En Synthesis lectures on games and computational intelligence. https://doi.org/10.1007/978-3-031-16719-5
    Gallotta, R., Arulkumaran, K., & Soros, L. B. (2022, May 12). Surrogate infeasible fitness acquirement fi-2pop for procedural content generation. arXiv.org. https://arxiv.org/abs/2205.05834 
    James Oxley - Matroid Theory (2011, Oxford University Press, USA) - libgen.lc
"""
#import sys
# sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps")
import random
import numpy as np
import Word_Stitching as ws
import re

"""## Samael  Search Aceleration via Machine Learning «Ceguera de Dios»
## https://www.youtube.com/watch?v=FHdlXe1bSe4
import torch
import torch.nn as nn
import torch.nn.funcytional as F
from torch.optim import SGD

import matplotlib.pylot as plt
import seborn as sns

class BasicNN(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, 6)
        self.pool = nn.MaxPool2d(3,3)
        
        self.fc2 = nn.Linear(9, 128)
        
    def Edges_Likelyhood(self, x, output_zise):
        x = self.pool(F.relu(self.conv(x)))
        x = F.relu(self.fc1(x))
        
        #Drop_min until output_zise
        
        
        return x   """
    
#defining word as a class and its components
class Word:
    all = []
    
    def __init__(self, Sequence):
        self.__Sequence = Sequence
        #Word.all = Word.all + [self]
        
    def OpenKey(self):
        if self.__Sequence:
            return self.__Sequence[0]
        else:
            return ""
    
    def CloseKey(self):
        if self.__Sequence:
            return self.__Sequence[-1]
        else:
            return ""
    
    def __repr__(self):
        return f"{self.__Sequence}"
    
    def __add__(self, other):
        return Word(self.__Sequence + other.__repr__())
    
    def __len__(self):
        return len(self.__Sequence)


#defining relations between words as an adjacency matrix
class Graph:
    def __init__(self, ExampleLevel, WordsList = []):
        self.Sequence = list(ExampleLevel)
        self.V = list(set(ExampleLevel))
        temp = np.empty((len(self.V),0))
        self.E = [list(x) for x in temp]
        del(temp)
        
        
        last_letter = Word(self.Sequence[0])
        Starter=True
        for letter in self.Sequence:
            letter=Word(letter)
            add_letter = True
            if Starter:
                Starter=False
            else: 
                for l in self.E[self.V.index(last_letter.__repr__())]:
                    if l.__repr__() == letter.__repr__():
                        add_letter = False
                        
                if add_letter:
                    self.E[self.V.index(last_letter.__repr__())] = self.E[self.V.index(last_letter.__repr__())]+[letter]
            
            last_letter=Word(letter)
            
        self.E = [list(set(x)) for x in self.E]
        
        for word in WordsList:
            word=Word(word)
            self.E[self.V.index(word.OpenKey())] = self.E[self.V.index(word.OpenKey())]+[word]

    # Print the graph
    def print_agraph(self):
        for i in self.V:
            print("Vertex " + i + ":", end="")
            print(self.E[self.V.index(i)])
            
    def Vertixes_of(self, v):
        return self.E[self.V.index(v)]

    def __len__(self):
        return len(self.V)
    
#Defining grammars
#receives an exmaple level as a string and an optional list of pre made levels
#pre made levels need to have same starting and final word the same as the example
class Grammar:
    
    all = []
    
    def __init__(self, ExampleLevel, WordsList = []): 
        
        self.__ExampleLevel = ExampleLevel 
        self.__WordsList = WordsList
        self.__Graph = Graph(self.__ExampleLevel, self.__WordsList)
            
        Grammar.all = Grammar.all + [self]
            
    def __repr__(self):
        return f"{self.__Graph}\n"
    
    def Add_Word(self, word):
        self.__WordsList = self.__WordsList + [word]
        self.__Graph = Graph(self.__ExampleLevel, self.WordsList)
        
    def Words_List(self):
        print(self.__WordsList)
    
    #gets all possible grammars of len 2
    def Simple_words(self):
        
        All_Keys = list(self.__ExampleLevel)
        Simple_Words = set()
        
        for key in All_Keys:
            x = self.__Graph.index[self.__Graph["from"] == key].tolist()[0]                
                
            for k in  All_Keys:
                y = self.__Graph.columns.get_loc(k)
                if self.__Graph.iloc[x,y] == 1:
                    Simple_Words = Simple_Words.union({key + k})
                            
        return #Simple_Words ##not yet implemented
    
    #receives a list of words and gives all the possible sublists for the especific Grammar
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
                
        return #Reducted_Blacklist ##not implemented yet
    
    #Generates a level of size N from a given grammar
    def N_Level_Generator(self, N, from_key = '', seed = None, module = 10, Gen_Game_mod=0):
        New_Level = Word("")
        random.seed(seed)
        #we don't talk about this part
        if N<= 0:
            return Word("")
        if N<=module:
            #from_keys with the level not being feasible
            Feasible_Level = False
            while not(Feasible_Level):
                #if not given a premade from_key, select at random
                if  from_key == '':
                    n = int(len(self.__Graph.V)*random.random())
                    New_Level = Word(self.__Graph.V[n])
                #check if from_key is in graph, else level can't be generated
                elif from_key in self.__Graph.V:
                    n = int(len(self.__Graph.Vertixes_of(from_key))* random.random())
                    new_Word = self.__Graph.Vertixes_of(from_key)[n]
                    New_Level = New_Level + new_Word
                else:
                    print("invalid from_key")
                    return
                while not(ws.Jumping_Fiasible_Word(New_Level.__repr__(),Game_mod=Gen_Game_mod)):
                    #print("no fiasible level")
                    n = int(len(self.__Graph.V)*random.random())
                    New_Level = Word(self.__Graph.V[n])
                #continue adding valid words to the final level until the end
                #you need to observe the relations in the graph
                while len(New_Level)<N:
                    n = int(len(self.__Graph.Vertixes_of(New_Level.CloseKey()))* random.random())
                    new_Word = self.__Graph.Vertixes_of(New_Level.CloseKey())[n]
                    while new_Word.OpenKey() == New_Level.CloseKey():
                        n = int(len(self.__Graph.Vertixes_of(New_Level.CloseKey()))* random.random())
                        new_Word = self.__Graph.Vertixes_of(New_Level.CloseKey())[n]
                    New_Level = New_Level + new_Word
                Feasible_Level = ws.Jumping_Fiasible_Word(New_Level.__repr__(),Game_mod=Gen_Game_mod)
            return New_Level
        else:
            for i in range(N//module):
                New_Level = New_Level + self.N_Level_Generator(module,New_Level.CloseKey(),module=module)
            New_Level = New_Level + self.N_Level_Generator(N%module,New_Level.CloseKey(),module=module)
            return New_Level
                
            


##TEst
G = Grammar("AABABBCDAEAEABABAABAAWSXMYZAAFFAAETEAAETEAASABASAAEETTEAETTEAAHIAAKLAAESEMEZE[[AAWAA")
Level = G.N_Level_Generator(20,"A").__repr__()
print(Level)