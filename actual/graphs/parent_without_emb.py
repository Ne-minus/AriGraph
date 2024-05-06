import os
import json
import requests
from time import sleep
import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cosine, euclidean

from utils import *
from prompts import *
from prompts_v2 import *
from parent_graph import TripletGraph

class GraphWithoutEmbeddings(TripletGraph):
    def __init__(self, model, system_prompt, threshold = 0.02):
        super().__init__(model, system_prompt, threshold)
        
    # For triplet without embeddings
    def str(self, triplet):
        return triplet[0] + ", " + triplet[2]["label"] + ", " + triplet[1]
    
    def get_all_triplets(self):
        return [self.str(triplet) for triplet in self.triplets]
    
    def delete_all(self):
        self.triplets, self.items = [], []
    
    # Filling graph
    def add_triplets(self, triplets):
        for triplet in triplets:
            if triplet[2]["label"] == "free":
                continue
            
            #if any(keyword in triplet[2]['label'] for keyword in ['south', 'north', 'east', 'west']) and "of" in triplet[2]['label'] :
            #    triplet[2]['label'] = 'is_' + triplet[2]['label']

            triplet = clear_triplet(triplet)
            if triplet not in self.triplets:
                self.triplets.append(triplet)
            if triplet[0] not in self.items:
                self.items.append(triplet[0])
            if triplet[1] not in self.items:
                self.items.append(triplet[1])       
                
    # Delete triplets exclude navigation ones            
    def delete_triplets(self, triplets, locations):
        for triplet in triplets:
            if triplet[0] in locations and triplet[1] in locations:
                continue
            if triplet in self.triplets:
                self.triplets.remove(triplet)
            
    # Associations by set of items. Step is a parameter for BFS
    def get_associated_triplets(self, items, steps = 2):
        items = deepcopy([string.lower() for string in items])
        associated_triplets = []
        
        for i in range(steps):
            now = set()
            for triplet in self.triplets:
                for item in items:
                    
                    if (item == triplet[0] or item == triplet[1]) and self.str(triplet) not in associated_triplets:
                        associated_triplets.append(self.str(triplet))
                        if item == triplet[0]:
                            now.add(triplet[1])
                        if item == triplet[1]:
                            now.add(triplet[0])    
                        
                        break
                    
            if "itself" in now:
                now.remove("itself")  
            items = now
        return associated_triplets
    
    def get_associated_triplets1(self, items, steps=2):
        items = set(item.lower() for item in items)  # Using a set for items, start with initial items
        associated_triplets = set()  # Using a set for efficient lookup and avoid duplicates

        for _ in range(steps):
            next_items = set()
            for triplet in self.triplets:
                for item in items:
                    if item == triplet[0]:
                        associated_triplets.add(self.str(triplet))
                        next_items.add(triplet[1])
                    elif item == triplet[1]:
                        associated_triplets.add(self.str(triplet))
                        next_items.add(triplet[0])

            if "itself" in next_items:
                next_items.remove("itself")
            items = next_items  # Update items for the next step

        return list(associated_triplets)
    
    # Exclude facts from 'triplets' which already in graph
    def exclude(self, triplets):
        new_triplets = []
        for triplet in triplets:
            triplet = clear_triplet(triplet)
            if triplet not in self.triplets:
                new_triplets.append(self.str(triplet))
                
        return new_triplets
    
    # Compute useful shape of graph with only spatial information
    def compute_spatial_graph(self, locations):
        locations = deepcopy(locations)
        if "player" in locations:
            locations.remove("player")
        graph = {}
        for triplet in self.triplets:
            if triplet[0] in locations and triplet[1] in locations and check_conn(triplet[2]["label"]):
                if triplet[0] in graph:
                    graph[triplet[0]]["connections"].append((triplet[2]["label"], triplet[1]))
                else:
                    graph[triplet[0]] = {"connections": [(triplet[2]["label"], triplet[1])]}
                
                if triplet[1] in graph:
                    graph[triplet[1]]["connections"].append(("reversed", triplet[0]))
                else:
                    graph[triplet[1]] = {"connections": [("reversed", triplet[0])]}
                    
        for loc in graph:
            connections = deepcopy(graph[loc]["connections"])
            connected_loc = [connection[1] for connection in connections if check_conn(connection[0])]
            for connection in connections:
                if connection[1] in connected_loc and connection[0] == "reversed":
                    graph[loc]["connections"].remove(connection)
        return graph
    
    def add_item(self, item):
        if item not in self.items:
            self.items.append(item)
        return item

    # Find shortest path between A and B if both in locations
    def find_path(self, a, b, locations):
        A = a.lower()
        B = b.lower()
        if A == 'Kids" Room':
            A = "Kids' Room"
        if B == 'Kids" Room':
            B = "Kids' Room"
        if A == B:
            return "You are already there"
        
        if A.lower() not in locations or B.lower() not in locations:
            return "Destination is unknown. Please, choose another destination or explore new paths and locations."
        spatial_graph = self.compute_spatial_graph(locations)
        current_set = {A}
        future_set = set()
        total_set = {A}
        found = False
        while len(current_set) > 0:
            for loc in current_set:
                for child in spatial_graph[loc]["connections"]:
                    if child[1] not in total_set:
                        future_set.add(child[1])
                        total_set.add(child[1])
                        spatial_graph[child[1]]["parent"] = loc
                        if child[1] == B:
                            found = True
                            break
                if found:
                    break
            if found:
                break
            current_set = future_set
            future_set = set()
        if not found:
            return "Destination isn't available according to had knowledges. Please, choose another destination or explore new paths and locations."
        path = []
        current_loc = B
        while current_loc != A:
            parent = spatial_graph[current_loc]["parent"]
            relation = find_relation(spatial_graph, parent, current_loc, True)
            path.append(relation)
            current_loc = parent
        return list(reversed(path))
    
    def print_graph(self):
    # Print all triplets
        print("Triplets in the graph:")
        for triplet_str in self.get_all_triplets():
            print(triplet_str)
        