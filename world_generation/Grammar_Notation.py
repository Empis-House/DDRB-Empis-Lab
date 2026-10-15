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
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
    
#defining word as a class and its components
class Word:
    all = []
    
    def __init__(self, Sequence):
        self._Sequence = Sequence
        #Word.all = Word.all + [self]
        
    def OpenKey(self):
        if self._Sequence:
            return self._Sequence[0]
        else:
            return ""
    
    def CloseKey(self):
        if self._Sequence:
            return self._Sequence[-1]
        else:
            return ""
    
    def __repr__(self):
        return f"{self._Sequence}"
    
    def __add__(self, other):
        return Word(self._Sequence + other._Sequence)
    
    def __len__(self):
        return len(self._Sequence)


#defining relations between words as an adjacency matrix
class Graph:
    def __init__(self, ExampleLevel, WordsList = [], additional_examples=None):
        sequences = [ExampleLevel.tolist()]
        sequences.extend(
            example.tolist() if hasattr(example, "tolist") else list(example)
            for example in (additional_examples or [])
        )
        self.Sequence = [token for sequence in sequences for token in sequence]
        self.V = list(dict.fromkeys(self.Sequence))
        self.E = [[] for _ in self.V]
        self._vertex_indices = {token: index for index, token in enumerate(self.V)}
        adjacency_keys = [set() for _ in self.V]

        for sequence in sequences:
            for source, destination in zip(sequence, sequence[1:]):
                source_index = self._vertex_indices[source]
                edge_key = (destination,)
                if edge_key not in adjacency_keys[source_index]:
                    edge = Word([destination])
                    neighbors = self.E[source_index]
                    neighbors.append(edge)
                    adjacency_keys[source_index].add(edge_key)

        for word in WordsList:
            edge = Word(word)
            source_index = self._vertex_indices[edge.OpenKey()]
            edge_key = tuple(edge._Sequence)
            if edge_key not in adjacency_keys[source_index]:
                neighbors = self.E[source_index]
                neighbors.append(edge)
                adjacency_keys[source_index].add(edge_key)

        self._non_self_successors = {
            token: [word for word in self.E[index] if word.OpenKey() != token]
            for token, index in self._vertex_indices.items()
        }

    # Print the graph
    def print_agraph(self):
        for i in self.V:
            print("Vertex " + str(i) + ":", end="")
            print(self.Vertixes_of(i))
            
    def Vertixes_of(self, v):
        return self.E[self._vertex_indices[v]]

    def Non_self_Vertixes_of(self, v):
        return self._non_self_successors[v]

    def __len__(self):
        return len(self.V)

    def out_degree_map(self):
        return {v: len(self.E[index]) for v, index in self._vertex_indices.items()}

    def in_degree_map(self):
        indegree = {v: 0 for v in self.V}
        for src in self.V:
            for dst_word in self.Vertixes_of(src):
                dst = dst_word.CloseKey()
                if dst in indegree:
                    indegree[dst] += 1
        return indegree
    
#Defining grammars
#receives an exmaple level as a string and an optional list of pre made levels
#pre made levels need to have same starting and final tile as the example level
class Grammar:
    
    all = []
    
    def __init__(self, ExampleLevel, knowledge, WordsList = [], additional_examples=None):
        
        self.__ExampleLevel = ExampleLevel 
        self.__WordsList = WordsList
        self.__AdditionalExamples = list(additional_examples or [])
        self.__Graph = Graph(self.__ExampleLevel, self.__WordsList, self.__AdditionalExamples)
        self.___Knowledge = knowledge
            
        Grammar.all = Grammar.all + [self]
            
    def __repr__(self):
        return f"{self.__Graph}\n"
    
    def Add_Word(self, word):
        self.__WordsList = self.__WordsList + [word]
        self.__Graph = Graph(self.__ExampleLevel, self.WordsList, self.__AdditionalExamples)
        
    def Words_List(self):
        return self.__WordsList
        
    def Knowledge(self):
        return self.___Knowledge

    def out_degree_tokens(self, min_degree=2):
        degree = max(0, int(min_degree))
        out_map = self.__Graph.out_degree_map()
        return [token for token, d in out_map.items() if d >= degree]

    def in_degree_tokens(self, min_degree=2):
        degree = max(0, int(min_degree))
        in_map = self.__Graph.in_degree_map()
        return [token for token, d in in_map.items() if d >= degree]

    def inner_degree_tokens(self, min_degree=2):
        """Tokens with both in-degree and out-degree >= min_degree."""
        degree = max(0, int(min_degree))
        out_map = self.__Graph.out_degree_map()
        in_map = self.__Graph.in_degree_map()
        return [
            token
            for token in self.__Graph.V
            if in_map.get(token, 0) >= degree and out_map.get(token, 0) >= degree
        ]
    
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
                            
        return #Simple_Words ##not implemented
    
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
    def N_Level_Generator(self, N, from_key = '', seed = None, module = 10, Gen_Game_mod=0, max_attempts=256):
        from Word_Stitching import Jumping_Fiasible_Word

        if N <= 0:
            return Word([])
        if from_key and from_key not in self.__Graph.V:
            raise ValueError(f"Invalid from_key '{from_key}' for current grammar graph")

        module = max(10, int(module))
        max_attempts = max(1, int(max_attempts))
        rng = random.Random(seed)

        if N <= module:
            if not self.__Graph.V:
                raise RuntimeError("Cannot generate a level from an empty grammar graph")

            for _ in range(max_attempts):
                if from_key:
                    start_words = self.__Graph.Vertixes_of(from_key)
                    if not start_words:
                        break
                    new_sequence = list(rng.choice(start_words)._Sequence)
                else:
                    new_sequence = [rng.choice(self.__Graph.V)]

                if not Jumping_Fiasible_Word(
                    new_sequence,
                    knowledge=self.___Knowledge,
                    Game_mod=Gen_Game_mod,
                ):
                    continue

                while len(new_sequence) < N:
                    successors = self.__Graph.Non_self_Vertixes_of(new_sequence[-1])
                    if not successors:
                        break
                    new_sequence.extend(rng.choice(successors)._Sequence)

                if len(new_sequence) == N and Jumping_Fiasible_Word(
                    new_sequence,
                    knowledge=self.___Knowledge,
                    Game_mod=Gen_Game_mod,
                ):
                    return Word(new_sequence)

            raise RuntimeError(f"Could not generate a feasible level of length {N} within {max_attempts} attempts")

        smaller_module = max(10, module // 2)
        for _ in range(max_attempts):
            new_sequence = []
            while len(new_sequence) < N:
                piece_length = min(module, N - len(new_sequence))
                piece_seed = rng.randrange(0, 2**32)
                try:
                    level_piece = self.N_Level_Generator(
                        piece_length,
                        from_key=new_sequence[-1] if new_sequence else '',
                        seed=piece_seed,
                        module=smaller_module,
                        Gen_Game_mod=Gen_Game_mod,
                        max_attempts=1,
                    )
                except RuntimeError:
                    break
                new_sequence.extend(level_piece._Sequence)

            if len(new_sequence) == N and Jumping_Fiasible_Word(
                new_sequence,
                knowledge=self.___Knowledge,
                Game_mod=Gen_Game_mod,
            ):
                return Word(new_sequence)

        raise RuntimeError(f"Could not generate a feasible level of length {N} within {max_attempts} attempts")

    def N_Level_Generator_Constrained(self,
                                      N,
                                      from_key='',
                                      allowed_end_keys=None,
                                      seed=None,
                                      module=10,
                                      Gen_Game_mod=0,
                                      max_attempts=256):
        """Generate a feasible level with optional start and end token constraints."""
        end_set = None
        if allowed_end_keys is not None:
            end_set = {k for k in allowed_end_keys if k in self.__Graph.V}
            if not end_set:
                end_set = None

        if from_key and from_key not in self.__Graph.V:
            raise ValueError(f"Invalid from_key '{from_key}' for current grammar graph")

        max_attempts = max(1, int(max_attempts))
        base_seed = None if seed is None else int(seed)

        for attempt in range(max_attempts):
            local_seed = None if base_seed is None else base_seed + attempt
            try:
                candidate = self.N_Level_Generator(
                    N,
                    from_key=from_key,
                    seed=local_seed,
                    module=module,
                    Gen_Game_mod=Gen_Game_mod,
                    max_attempts=1,
                )
            except RuntimeError:
                continue
            if end_set is None or candidate.CloseKey() in end_set:
                return candidate

        raise RuntimeError(
            "Could not generate constrained level with requested start/end grammar tokens"
        )
                