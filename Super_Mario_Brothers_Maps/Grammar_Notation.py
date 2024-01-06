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
"""
import sys
sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps")
import random
import numpy as np
import Word_Stitching as ws

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
    

class Word:
    all = []
    
    def __init__(self, Sequence):
        self.__Sequence = Sequence
        #Word.all = Word.all + [self]
        
    def OpenKey(self):
        return self.__Sequence[0]
    
    def CloseKey(self):
        return self.__Sequence[-1]
    
    def Tail_Condition(self,n=3):
        
        if len(self.__Sequence)<n:
            return True
        
        List = list(self.__Sequence[-n:])
        
        if(List[0]==List[1])and(List[2]==List[1]):
            return False
        else:
            return True
    
    def __repr__(self):
        return f"{self.__Sequence}"
    
    def __add__(self, other):
        return Word(self.__Sequence + other.__repr__())
    
    def __len__(self):
        return len(self.__Sequence)


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
class Grammar:
    
    all = []
    
    def __init__(self, ExampleLevel, WordsList = []): 
        
        self.__ExampleLevel = ExampleLevel 
        self.__WordsList = WordsList
        self.__Graph = Graph(self.__ExampleLevel, self.__WordsList)
            
        Grammar.all = Grammar.all + [self]
            
    def __repr__(self):
        return f"{self.__Keys_Matrix}\n"
    
    def Add_Word(self, word):
        self.__WordsList = self.__WordsList + [word]
        self.__Graph = Graph(self.__ExampleLevel, self.WordsList)
        
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
    
    def N_Level_Generator(self, N, Start = '', seed = None, module = 10):

        random.seed(seed)
        if False:
            New_Level = Word("")
            # N = n*module + k
            k = N%module     
            m = int(N/module)
            
            Feasible_Level = False
            
            if  Start == '':
                n = int(len(self.__Graph.V)*random.random())
                New_Level = Word(self.__Graph.V[n])
            elif Start in self.__Graph.V:
                New_Level = Word(Start)
            else:
                print("invalid start")
                return
            while not(ws.Jumping_Fiasible_Word(New_Level.__repr__())):
                #print("no fiasible start")
                n = int(len(self.__Graph.V)*random.random())
                New_Level = Word(self.__Graph.V[n])
            for i in range(m):
                n = int(len(self.__Graph.Vertixes_of(New_Level.CloseKey()))* random.random())
                new_Start = self.__Graph.Vertixes_of(New_Level.CloseKey())[n]
                new_Word = self.N_Level_Generator(module,Start = new_Start)
                New_Level = New_Level + new_Word
            Feasible_Level = ws.Jumping_Fiasible_Word(module,New_Level.__repr__())
                
            return New_Level
            
            return New_Level
        
        elif N>0:
            Feasible_Level = False
            while not(Feasible_Level):
                if  Start == '':
                    n = int(len(self.__Graph.V)*random.random())
                    New_Level = Word(self.__Graph.V[n])
                elif Start in self.__Graph.V:
                    New_Level = Word(Start)
                else:
                    print("invalid start")
                    return
                while not(ws.Jumping_Fiasible_Word(New_Level.__repr__())):
                    #print("no fiasible start")
                    n = int(len(self.__Graph.V)*random.random())
                    New_Level = Word(self.__Graph.V[n])
                while len(New_Level)<N:
                    n = int(len(self.__Graph.Vertixes_of(New_Level.CloseKey()))* random.random())
                    new_Word = self.__Graph.Vertixes_of(New_Level.CloseKey())[n]
                    while new_Word.OpenKey() == New_Level.CloseKey():
                        n = int(len(self.__Graph.Vertixes_of(New_Level.CloseKey()))* random.random())
                        new_Word = self.__Graph.Vertixes_of(New_Level.CloseKey())[n]
                    New_Level = New_Level + new_Word
                Feasible_Level = ws.Jumping_Fiasible_Word(New_Level.__repr__())
            return New_Level

G = Grammar("AABABBCDAEAEABABAABAAWSXMYZAAFFAAETEAAETEAASABASAAEETTEAETTEAAHIAAKLAAESEMEZE[[AAWAA")
Level = G.N_Level_Generator(20).__repr__()
print(Level)