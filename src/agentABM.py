

# Parent Class for Agents, which can be used to create different types of agents
# such as GridAgent, NetAgent, etc.

import networkx as nx
from itertools import product
import numpy as np

##########################################
#### AGENT (parent class)####
##########################################

class ABMAgent:
    def __init__(self, config, id=None, state=None):
        # Validate config parameter
        required_keys = ["parameters"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Config is missing required key: {key}")
        self.config = config
        self.id = id
        self.state = state
        self.message = ""
        self.recent_memory = []
        self.memory = []
        
    def perceive(self, agents, global_perception, k=1):
        """
        Default perception behavior, can be overridden by subclasses.
        By default, agents perceive the state of their neighbors at distance k and the global state.
        
        Parameters:
        agents (dict): A dictionary of agents with their positions as keys.
        global_perception (str): The global state perception.
        k (int): The distance to consider for local perception.
        
        Returns:
        dict: A dictionary containing local and global perceptions.
        """
    def update(self, perception,step,  **kwargs):
        """
        return perception
        Returns:
        None
        """
        # Placeholder for update logic, to be overridden by subclasses. 
        # By default, the agent does not update its state but its memory. Static agent
        perception = " ".join(perception["local"]) + perception["global"]
        self.update_recent_memory(perception)
        self.update_external_memory(perception)
        
    def transmit(self, perception, step=None, **kwargs):
        """
        Default transmit behavior, can be overridden by subclasses.
        
        Parameters:
        perception (dict): A dictionary containing local and global perceptions.
        kwargs: Additional keyword arguments.
        
        Returns: 
        """
        return perception

    def update(self, perception,step, **kwargs):
        # Placeholder for update logic, to be overridden by subclasses. 
        # By default, the agent do not update its state but its memory. Static agent
        perception = " ".join(perception["local"]) + perception["global"]
        self.update_recent_memory(perception)
        self.update_external_memory(perception)

    def transmit(self, perception,step, **kwargs):
        # Placeholder transmit behavior, can be overridden by subclasses
        self.message = self.get_state_as_text()


    def get_state_as_text(self):
        """
        Return the textual state of the agent (by default the state itself)
        """
        return str(self.state)
    
    def extract_state_from_text(self, text):
        """
        Return the state from a textual form
        """
        return text
    
    def forget(self):
        """
        By default, forget randomly erase an element from the memory with a certain probability stated in config
        """
        if self.forgetting_rate>0:
            if len(self.memory) > 0:
                if np.random.rand() < self.forgetting_rate:
                    self.memory.pop(np.random.randint(len(self.memory)))
    
    def update_external_memory(self, memory):
        """
        By default, save the memory in the external memory and forget one element
        """
        self.memory.append(memory)
        self.forget()

    def update_recent_memory(self, memory):
        """
        Update the recent memory list with the most recent memory.
        Ensures that the list is capped at m items, removing the oldest memory if necessary.
        
        Parameters:
        memory (str): The memory to be added to the recent memory list.
        
        Returns:
        None
        """
        if len(self.recent_memory) >= self.memory_buffer_size:
            self.recent_memory.pop(0)  # Remove the oldest memory
        self.recent_memory.append(memory)  # Add the new memory

##########################################
#### GRID AGENT ####
##########################################
# ie GridAgent is a class that can be used to create agents which are part of a grid


class GridABMAgent(ABMAgent):
    def __init__(self, config, id=None, position=None, state=None):
        super().__init__(config, id, state)
        self.position = tuple(position) if position is not None else None  # Ensure position is a tuple for immutability

    def get_neighbors(self, agents, k=1):
        offsets = list(product(range(-k, k+1), repeat=len(self.position)))
        offsets.remove((0,) * len(self.position))
        neighbors = []
        for offset in offsets:
            neighbor_pos = tuple(self.position[i] + offset[i] for i in range(len(self.position)))
            if neighbor_pos in agents:
                neighbors.append(agents[neighbor_pos])
        return neighbors



##########################################
#### NET AGENT ####
##########################################
        
# ie NetAgent is a class that can be used to create agents which are part of a network

class NetABMAgent(ABMAgent):
    def __init__(self, config, id=None, state=None, network=None):
        super().__init__(config, id, state)
        self.network = network

    def get_neighbors(self, network, k=1):
        all_neighbors = nx.single_source_shortest_path_length(network, self.id, cutoff=k)
        all_neighbors.pop(self.id, None)
        return list(all_neighbors.keys())
