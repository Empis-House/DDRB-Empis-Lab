# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:00:20 2023
Check List:
    *More Functions
    *Mapping Functions frome Grids
@author: Juan J. Rueda M.
"""
""""""
# import sys
# sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps")

import scripts.Grammar_Notation as gn
import scripts.Word_Stitching as ws
import scripts.Performance_Evaluation as pe
import pandas as pd
import numpy as np
from itertools import product
import time
import numpy as np
from scipy.spatial import distance


#CReate Map Elite Class
#Good explanation on https://arxiv.org/abs/2205.05834
class Map_Elite:
    all = []
    
    def __init__(self, function, alphas =[0], Variety_Dominess = [], Grid_points =[]):
        # function must acept a Level as input
        # fuction must have Image the real interval [0,1]
        # The Filtter will capting the optimal value (min)
        self.__N = len(alphas)
        self.__f = function
        self.__alphas = alphas
        self.__Variety_Dominess = Variety_Dominess
        
        self.__Opt_Values = None
        self.__Optimal_strings = None
        
        self.__Grid_points = Grid_points
        
        if not(self.__Variety_Dominess):
            self.__Opt_Values = np.ones(self.__N)
            self.__Optimal_strings = [""]*self.__N
            
        elif len(self.__Variety_Dominess) == len(self.__Grid_points):
            self.__Opt_Values, self.__Optimal_strings = self.transform_lists_to_dict(self.__Grid_points,self.__N)
            
        else:
            print("The Variety Grid failse, the Map have set as Default")
            self.__Opt_Values = np.ones(self.__N)
            self.__Optimal_strings = [""]*self.__N
    
    def __repr__(self):
        return str([self.__Optimal_strings,self.__Opt_Values])
    
    def __len__(self):
        return self.__N
    
    def transform_lists_to_dict(self, lists, N):
        # Using itertools.product to get all combinations and converting to a list
        keys = list(product(*lists))
        
        # Creating the dictionaries
        transformed_dict = {tuple(key): np.ones(N) for key in keys}
        transformed_dict2 = {tuple(key): ['']*N for key in keys}
        
        return transformed_dict, transformed_dict2

    def get_nearest_label(self,input_tuple, dictionary):
        # Initialize the minimum distance to a large number
        min_dist = float('inf')
        nearest_key = None
        
        # Iterate through the dictionary keys
        for key in dictionary.keys():
            # Calculate the Euclidean distance between input_tuple and the current key
            dist = distance.euclidean(input_tuple, key)
            
            # Update nearest_key if the current distance is smaller than the minimum distance
            if dist < min_dist:
                min_dist = dist
                nearest_key = key
        
        return nearest_key

    def apply_functions(self,functions, x, df):
        """
        Apply a list of functions to an input x and return the results as a tuple.

        :param functions: List of functions [f1, f2, f3]
        :param x: The input to the functions
        :return: A tuple of results (f1(x), f2(x), f3(x))
        """
        return tuple(f(x,df=df) for f in functions)   
            
    def Quest(self, Level,df):
        y = self.__f(Level,df=df)
        
        if not(self.__Variety_Dominess):
            for i in range(self.__N):
                if self.__Optimal_strings[i]=="" or np.abs(self.__alphas[i]-y) < np.abs(self.__alphas[i]-self.__Opt_Values[i]):
                    self.__Opt_Values[i] = y
                    self.__Optimal_strings[i] = Level
            return
        

        input_tuple = self.get_nearest_label(self.apply_functions(self.__Variety_Dominess, Level,df=df), self.__Opt_Values)
        for i in range(self.__N):
            if self.__Optimal_strings[input_tuple][i]=="" or np.abs(self.__alphas[i]-y) < np.abs(self.__alphas[i]-self.__Opt_Values[input_tuple][i]):
                self.__Opt_Values[input_tuple][i] = y
                self.__Optimal_strings[input_tuple][i] = Level
        return
    
    def format_array(self,arr, precision=2):
        return np.round(arr, precision)

    def printable_formatted_dict(self,d, precision=2):
        report = []
        for key, value in d.items():
            rounded_value = self.format_array(value, precision)
            report.append({key: rounded_value})
        return report
    
    def Report(self, Type = "Values"):
        if Type == "Values":
            return self.printable_formatted_dict(self.__Opt_Values)
        if Type == "Strings":
            return self.__Optimal_strings
        else:
            return
    
    def Structure(self, row):
        return "".join(row[2:])

    def Key_substraction(self, x, df):
        "-------------X"
        "---------XXXXX"
        "--------------"
        "-------XXXXXXX"
        if len(df[df["Structures"] ==  x]["token"])==0:
            return "?"
        else:
            return np.array(df[df["Structures"] ==  x]["token"])[0]

    def Extract_Level_String(self, examples_codes,df):
        for example_code in examples_codes:
            Temporal = pd.DataFrame(columns=["Structures","token"] + ["{}".format(i) for i in range(13)])
            with open(r"Super_Mario_Brothers_Maps/Processed/mario-{}.txt".format(example_code)) as infile:
                i = 0
                for line in infile: 
                    Temporal["{}".format(i)] = list(line.split()[0])
                    i+=1
            Level = pd.DataFrame(columns=["Structures","token"] + ["{}".format(i) for i in range(13)])
            Level = pd.concat([Level,Temporal], axis=0)
                
        Level["Structures"] = Level.apply(self.Structure, axis=1)
        
        for i in range(len(Level["Structures"])):
            Level["token"][i]= self.Key_substraction(Level["Structures"][i], df=df)
            
        return list(Level["token"])
    
    
    def Generate_Mapping(self,example_code, Knowledge, df, Level_Len, module=None, batch_size=1, epochs=256, seed=None,Game_Mode = 0):
        if not(module):
            module = Level_Len

        level_sec = self.Extract_Level_String([example_code[0]],df=df) #+" "+ Extract_Level_String([example_code[1]])
        #Display_Level(Extract_Level_String([example_code[0]]),level_name="1-1".format(example_code),df=df)
        #Display_Level(Extract_Level_String([example_code[1]]),level_name="1-2".format(example_code),df=df)

        G = gn.Grammar(level_sec,knowledge = Knowledge)
        pre_Level = G.N_Level_Generator(Level_Len,module=module,Gen_Game_mod=Game_Mode,seed=seed)._Sequence
        while not(ws.Jumping_Fiasible_Word(pre_Level, knowledge = Knowledge, Game_mod=Game_Mode)):
            pre_Level = G.N_Level_Generator(Level_Len,module=module,Gen_Game_mod=Game_Mode,seed=seed)._Sequence
        self.Quest(pre_Level,df=df)


        for j in range(batch_size):
            print("Start",Level_Len)
            print("XX", "t", "Time", "--", "Performance", "--","--", "Fiasible_Word %")
            base_time = time.time()
            t=0
            T=0
            for i in range(epochs):
                pre_Level = G.N_Level_Generator(Level_Len,module=module,Gen_Game_mod=Game_Mode,seed=seed)._Sequence
                T+=1
                while not(ws.Jumping_Fiasible_Word(pre_Level, knowledge = Knowledge, Game_mod=Game_Mode)):
                    T+=1
                    pre_Level = G.N_Level_Generator(Level_Len,module=module,Gen_Game_mod=Game_Mode,seed=seed)._Sequence
                
                t+=1
                self.Quest(pre_Level,df=df)
                
                    
                if i%100 == 0 or i == batch_size-1:
                    delta_time = time.time() - base_time
                    print(i, "t", "%.2f" % (delta_time/60), "--", self.Report(), "--", "%.2f" % (t/T*100),"%")

    #display and save final level if name is included
    def Display_Level(self,Knowledge, Level, df, level_name=None):

        if Level=="":
            print("Gap")
            return

        print(ws.Jumping_Fiasible_Word(Level,knowledge = Knowledge))

                
        Display = pd.DataFrame(columns= ["{}".format(i) for i in range(14)])
        for Key in list(Level):
            new_row = list(list(df.loc[df["token"] == Key,"Structures" ])[0])
            Display.loc[len(Display)] = new_row
            
            
        Display = Display.transpose()
        if level_name != None:
            Display.to_csv(r'Super_Mario_Brothers_Maps/final_levels/{}.txt'.format(level_name), index=False,header=False,sep=",")
        
        print(Display)   

    def find_exact_matches(self,reference, quest):
        # Create a dictionary to map each value in the reference list to its index
        reference_index_map = {value: idx for idx, value in enumerate(reference)}
        
        # List to store the matches
        matches = []
        
        # Iterate over each element in the quest list
        for q in quest:
            # Check if the element is in the reference map and append its index if it is
            if q in reference_index_map:
                matches.append(reference_index_map[q])
        
        return matches

    def Plot_Coordinate(self, cartesian_list, level_curves,Knowledge,df,level_name=None):
        indexes = self.find_exact_matches(self.__alphas,level_curves)
        for x in cartesian_list:
            near_x = self.get_nearest_label(x,self.__Optimal_strings)
            for i in indexes:
                self.Display_Level(Knowledge, self.__Optimal_strings[near_x][i], df, level_name)



