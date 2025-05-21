
import numpy as np
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.agent import GridLLMAgent
from src.prompts.persona import NAMES
import random

class SchellingAgent(GridLLMAgent):
    # TODO DOUBLE INHERITANCE LLM AGENT AND GRID AGENT...
    def __init__(self, config, position=None, state=None, degree= None, client=None, model_name=None, cached_probabilities=None):
        """
        Note here the state of the Schelling agent is the persona index (0 or 1) from which generate the system prompt of the agent.
        It is static.
        """
        self.name = random.choice(NAMES)
        system_prompt_extra = ["From now on, you have to realistically play the role of a human named {name}.", ""]
        self.persona_prompt = self.get_persona_prompt_from_persona(config, state, degree)
        self.ingroup_id = self.persona + "-" + str(degree) if degree is not None else self.persona
        self.cached_probabilities = cached_probabilities[self.ingroup_id] if cached_probabilities is not None and self.ingroup_id in cached_probabilities else None
        super().__init__(config, position=position, state=state, persona_prompt=self.persona_prompt,  name=self.name, system_prompt_extra=system_prompt_extra,model_name=model_name, client=client)

        self.message = self.persona_prompt # here message is the state of the agent
        assert self.config["parameters"]["moving_mode"] in ["random", "biased", "optimal", "desired"], "Moving mode must be either random, biased or optimal."
        
    def get_persona_prompt_from_persona(self, config, state, degree):

        """
        Generate a persona prompt for a given name and persona type and degree of intensity.
        Args:
            name (str): The name of the persona.
            persona (int): The index of the persona type
            degree (int): The degree of intensity (1-3) of the persona.
        Returns:
            str: The generated persona prompt.
        """
    
        self.persona = config["parameters"]["personas"][state]
        assert 0 <= state <= len(self.persona), "State must be between 0 and num persona authorised."
        if degree is not None:
            assert 1 <= degree <= 3, f"Degree must be between 1 and 3, not {degree}."
        from src.models.schelling.prompts.persona import PERSONAS, DEGREES
        if self.persona in PERSONAS:
            if degree:
                prompt = PERSONAS[self.persona].format(name=self.name, adjective=DEGREES[degree-1]["adjective"], adverb=DEGREES[degree-1]["adverb"])
            else:
                prompt = PERSONAS[self.persona].format(name=self.name, adjective="", adverb="").replace("  ", " ")
            return prompt.format(name=self.name)
        else:
            raise ValueError("Persona type not found.")
    
    def get_state_as_text(self):
        """
        Return the textual state of the agent
        """
        return self.persona_prompt #TODO

    def check_similarity(self, neighbor):

        return self.persona == neighbor.persona
    
    def get_probabilities_cached(self, neighbors_id):
        """
        Get cached probabilities for a given set of neighbors
        """
        #Beware, because it is a set, needs to try out different combinations
        if len(neighbors_id)==0:
            return 0 #TODO: no nieghbor case
        all_keys = [key.split("_") for key in self.cached_probabilities.keys()]
        for key in all_keys:
            if set(key) == set(neighbors_id):
                key_="_".join(key)
                if key_ in self.cached_probabilities and self.cached_probabilities[key_]['probability'] is not None and (self.cached_probabilities[key_]['MOVE']+self.cached_probabilities[key_]['STAY'])==0:
                    print(f"DEBUG detail: BEWARE not tested for agent {self.ingroup_id} and key {key_}")
                    return None
                if key_ in self.cached_probabilities and self.cached_probabilities[key_]['probability'] is not None and (self.cached_probabilities[key_]['MOVE']+self.cached_probabilities[key_]['STAY'])>0:
                    p = self.cached_probabilities[key_]['probability']
                    if p>0:
                        if self.config["debug"]:
                            print(f"DEBUG detail: Found cached proba {p} >0 for agent {self.ingroup_id} and key {key_}")
                    else:
                        assert self.cached_probabilities[key_]["MOVE"] == 0 and  self.cached_probabilities[key_]["STAY"] >0
                    return p
        if self.config["debug"]:
            print(f"DEBUG detail: No cached proba for agent {self.ingroup_id} and key {neighbors_id}") 
        return None

    def get_cached_data(self, neighbors):
        assert self.cached_probabilities is not None, "Cached probabilities must be provided."
        if self.cached_probabilities is not None:
            neighbors_id = [n.ingroup_id for n in neighbors]
            p = self.get_probabilities_cached(neighbors_id)
            return p
                
        return None
    
    def perceive(self, agents, global_perception=None, step=None):
        """
        #NOTE: Most of the procedure is the same as the parent class, appart the cached data.
        """

        perception = super().perceive(agents, global_perception, step =step)
        
        # Added #TODO: integrate parent method
        neighbors = self.get_neighbors(agents, k=self.config["parameters"]["perception_radius"])
        self.score = (
            1 if len(neighbors) == 0 else sum([1 for n in neighbors if self.check_similarity(n)]) / len(neighbors)
        )
        perception["cache"] = self.get_cached_data(neighbors)
        return perception

    def if_random_action(self, activated_noise):
        #len(list(possible_actions.keys())) == 0 or 
        if self.config["parameters"]["moving_mode"]=="random" or activated_noise:
            if self.config["debug"]:
                print("DEBUG detail: Random Move")
            return True
        return False
    
    def get_desired_actions(self, rated_positions):
        # filter dictionary to only keep better positions 
        if self.config["parameters"]["moving_mode"] in ["random", "biased"]:
            desirable_positions = {k: v for k, v in rated_positions.items() if k != self.position}
        elif self.config["parameters"]["moving_mode"] in ["optimal", "desired"]:
            desirable_positions = {k: v for k, v in rated_positions.items() if k != self.position and v[self.state] > self.score}
        else:
            raise ValueError("Moving mode must be either random, biased or optimal.")
        return desirable_positions
    
    def update(self, perception, step = None, rated_positions=None):
        """
        The agent may decided to move to a new position based on its perception.
        """
        assert perception["cache"] is not None #TODO: TEMP  
        if perception["cache"] is not None:
            return self.update_cache(perception, rated_positions)
        
        context = perception["reflected"] if perception["reflected"] else perception["aggregated"]
        
        if perception is None:
            if self.config["debug"]:
                print(f"DEBUG detail: No perception for agent {self.name}!")
            return None, None
        
        activated_noise = False
        if np.random.rand() < self.config["parameters"]["noise"]:
            action = np.random.choice([0, 1])
            activated_noise = True
        else: # Ask LLM
            prompt = self.build_update_prompt(context, step =step)
            response = self.ask_llm(prompt, max_tokens=5).strip()      
            action = self.turn_response_into_action(response, step=step)
        
        # Decided for no action
        if action == 0:
            return None, None
        
        # Decide specific action
        new_position = self.pick_new_position(rated_positions, activated_noise = activated_noise)
        if new_position is None:
            return None, None
        self.position = new_position
        
        #### 4 -- Save Historics
        self.trace["state"].append(self.state)
        #self.trace["message"].append(self.message)
        
        return 1, self.position
    
    def update_cache(self, perception, rated_positions=None):
        noise = self.config["parameters"]["noise"]
        p_move = perception["cache"]
        p_effective = (1 - noise) * p_move + noise * np.random.rand()
        action = 0 if np.random.rand() > p_effective else 1   
        # biased towards moving (since people move sometimes beyond ideology): p_effective = (1 - noise) * p_move + noise
        # pure randomness p_effective = (1 - noise) * p_move + noise * np.random.rand()
        if action == 0:
            return None, None
        self.position = self.pick_new_position(rated_positions, activated_noise = False)
        return 1, self.position
            

    def pick_new_position(self, rated_positions, activated_noise = False):
        
        desired_positions = self.get_desired_actions(rated_positions)

        # with a bit of noise, will choose randomly the position too
        if self.if_random_action(activated_noise):
            if self.config["debug"]:
                print("DEBUG detail: Random Move")
            positions = list(rated_positions.keys())
            return positions[np.random.choice(len(positions))]
        elif len(desired_positions) == 0:
            if self.config["debug"]:
                print("DEBUG detail: !No Desired Positions!")
            return None
        # If unsatisfied, move to empty house
        if self.config["debug"]:
            print("DEBUG detail: !Selected Move!")
        if self.config["parameters"]["moving_mode"]=="desired":
            weights = [1 for v in desired_positions.values()]  # similar ratio to sample neighborhood
        elif self.config["parameters"]["moving_mode"] in ["optimal", "biased"]:
            weights = [v[self.state] for v in desired_positions.values()]
        else:
            raise NotImplementedError("Moving mode must be either random, biased or optimal or desired.")
        desired_positions = list(desired_positions.keys())
        new_index = np.random.choice(len(desired_positions), p=np.array(weights) / sum(weights))
        return desired_positions[new_index]     
        
        
    def turn_response_into_action(self, response, step=None):
        """
        Turn the response of an LLM into an action. This is a placeholder!
        """
        if response is None:
            return 0
        if ("STAY" in response and "MOVE" in response) or (("STAY" not in response) and ("MOVE" not in response)):
            print(f"\(°Ω°)/ WARNING \(°Ω°)/ issue with response do not contain what it should --agent stay static: {response}")
            return 0
        
        if response!="STAY" and response!="MOVE":
            print(f"¯\(°_o)/¯ BEWARE, response not perfectly well formatted: {response}")

        if ("STAY" in response):
            return 0
        
        assert "MOVE" in response
            
        return 1

    def build_update_prompt(self, context, step = None):
        """
        Build the update prompt
        """
        prompt = context + self.META_PROMPTS["update"].format(name=self.name)
         #TODO FILL IN PROMPT for AGENT LEVEL more generally

        if len(self.config["parameters"]["inject_prompt_bias"])>0:
            for bias in self.config["parameters"]["inject_prompt_bias"]:
                prompt = prompt + self.config["prompts"]["bias"][bias].format(persona=self.persona)
        
        return prompt

            
        

