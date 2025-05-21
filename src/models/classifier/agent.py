
import numpy as np
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.agent import GridLLMAgent
from src.prompts.persona import NAMES
import random
import numpy as np
from typing import List, Tuple
from pydantic import BaseModel

class Hypothesis(BaseModel):
    digit: int
    self_role: str
    confidence: int #in percentage
    comment: str

    def __str__(self) -> str:
        return f"Shape {self.digit}, {self.self_role}, with confidence {self.confidence}%, Comment: {self.comment}"
        
class UpdateFormat(BaseModel): #TODO> FOR Schelling
    analysis: str
    observations: List[str]
    hypotheses: List[Hypothesis] 
    
class InitialFormat(BaseModel): 
    analysis: str
    observation: str
    hypotheses: List[Hypothesis] 

class SelfClassifyingAgent(GridLLMAgent):
    def __init__(self, config, position=None, model_name =None, client=None):
        self.name = random.choice(NAMES) #TODO: REMOVE not care
        
        #TODO: CHANGE TERMINOLOGY into visible_state and not. or 0 component state visible
        
        #Visible state
        if config["parameters"]["initial_belief"]=="none":
            self.state = None
            self.state_confidence = None
        elif config["parameters"]["initial_belief"] == "random":
            self.state = random.choice(config["shapes"])
            self.state_confidence = None
        else:
            self.state = config["parameters"]["initial_belief"]
            self.state_confidence = config["parameters"]["initial_confidence"]
        assert self.state is None or self.state in config["shapes"], f"Initial belief {self.state} should be in {config['shapes']}"
        
        #Hidden state
        self.observations = [] # list of observations
        self.beliefs = [] #list hypotheses
        grid_size = (config["grid_size"][0], config["grid_size"][1])
        
        self.persona_prompt = config["prompts"]["system_prompt"].replace("#shapes", str(config["shapes"])).replace("#max_hypotheses", str(config["parameters"]["max_hypotheses"])).replace("#grid_size", str(grid_size))
        #TODO Automate from config file
        
        super().__init__(config, position=position, state=self.state, persona = str(self.state), persona_prompt=self.persona_prompt, model_name = model_name, name=self.name, client=client)

        self.message = ""
    
    def initialisation(self, agents):
        """
        Initialize the hypotheses of the agent based on the other agents.
        """
        # First observation neighborhood withtout messages
        neighbors_positions = self.collect_neighbor_positions(agents)
        assert len(neighbors_positions) > 0, f"Agent {self.name} has no neighbors, the shape should be connected!"
        self_position = f"You are situated at {self.position} on a {self.config['grid_size']} grid."
        if self.config["perception_with_position"]:
            context = self_position + f"You have {len(neighbors_positions)} neighbors situated at the following positions: " + "and ".join(neighbors_positions)
        elif self.config["perception_with_orientation"]:
            context = self_position + f"You have {len(neighbors_positions)} neighbors situated, relatively to you, at the following positions: " + "and ".join(neighbors_positions)  
        else: #TODO: REMOVE THIS POSSIBILKIT
            context = f"You have {len(neighbors_positions)} neighbors situated, relatively to you, at the following positions: " + "and ".join(neighbors_positions)  
        # Form first hypotheses and state
        prompt = self.build_initialization_prompt(context)
        response = self.ask_llm(prompt, max_tokens=2000, response_format = InitialFormat)
        
        if response:
            if self.config["debug"]:
                print(f"DEBUG detail: Agent {self.name} got asked {prompt}.")
            action = self.turn_response_into_action(response, step = 0)
            self.trace["state"].append(self.state)
            hidden_state = (
                "Observations: " + ", ".join(self.observations) +
                ". Beliefs: " + ", ".join(str(belief) for belief in self.beliefs)
            )
            self.trace["hidden_state"].append(hidden_state) #all the thoughts the agent has transmitted
            self.trace["message"].append(self.message) #all the messages the agent has transmitted
        else:
            print(f"DEBUG None response from prompt", prompt)
        
    

    def get_self_perception(self, only_observations=False):
        self_prompt=""
        # Add observations
        if self.config["perception_with_position"] or self.config["perception_with_orientation"]:
            grid_size = (self.config["grid_size"][0], self.config["grid_size"][1])
            self_prompt += f"You are placed at {self.position} on a {grid_size} grid.\n"
        if len(self.observations)>0:
            self_prompt += f"You gathered the following observations until now: {self.observations}.\n"
        if not only_observations and len(self.beliefs)>0: #TODO Say reflection this
            self_prompt += f"Your current hypotheses are:" + ".".join(
                f"H{i+1}: {h}"  
                for i, h in enumerate(self.beliefs)
            )
        return self_prompt
         
        
    def perceive(self, agents, global_perception=None, step=None):
        """
        The agent perceives its own state, and hidden state and messages from the other agents.
        The agent also perceive the state of the other agents.
        """
        
        perception = super().perceive(agents, global_perception, step=step)       
        #if self.config["debug"]:
        #    print(f"DEBUG Detail: Aggregated Perception", perception["aggregated"])
        return perception
    
    def update(self, perception, step=None, **kwargs):
        """
        The agent may decided to update its belief based on the perception.
        """
        context = perception["aggregated"] #perception["reflected"] if (perception["reflected"] and len(perception["reflected"])>5) else 
        
        if perception is None:
            if self.config["debug"]:
                print(f"DEBUG detail: No perception for agent {self.name}!")
            return 0, None
    
        
        # Ask LLM
        shapes = self.config["shapes"]
        prompt = self.build_update_prompt(context, step = step) + f"Reminder: The shape is a digit among {shapes}, and it may have a small thickness."
        response = self.ask_llm(prompt, max_tokens=3000, response_format = UpdateFormat)
        if response:
            #if self.config["debug"]:
            #    print(f"DEBUG detail: Agent {self.name} got asked {prompt}.")
            action = self.turn_response_into_action(response, step = step)
        else:
            print(f"DEBUG None response from prompt", prompt)
            action = 0

        #### 4 -- Save Historics
        self.trace["state"].append(self.state)
        hidden_state = (
            "Observations: " + ", ".join(self.observations) +
            ". Beliefs: " + ", ".join(str(belief) for belief in self.beliefs)
        )
        self.trace["hidden_state"].append(hidden_state) #all the thoughts the agent has transmitted
        self.trace["message"].append(self.message) #all the messages the agent has transmitted
         
        return action, None
  
      
    def build_update_prompt(self, context, step = None):
        """
        Build the update prompt
        """
        
        step_info = f"[Step {step} / {self.config['max_steps']}]" if step is not None else ""
        prompt = step_info + context + self.META_PROMPTS["update"].format(name=self.name)
        if self.config["parameters"]["update_final_prompt"] and self.config["max_steps"] - step <= 3:
            prompt += self.config["prompts"]["update_final"] #.replace("#current_consensus", current_consensus) #TODO may add current state consensus?
        
        return prompt
  
    def transmit(self, perception, step = None): #TODO context
        
       
        prompt = self.META_PROMPTS["transmission"]
        if step == 0:
            context = self.get_self_perception(only_observations=True) #self perception and self reflections here ! #UPDATED AFTER UPDATE !!
        else:
            context = self.get_self_perception()
            
        #NOTE: May give here more information such as previous message transmitted etc #TODO: Function
        response = self.ask_llm(context + prompt,  max_tokens=300)  # max_tokens=100
        if self.config["debug"]:
            print(f"DEBUG TRANSMIT response of {self.name}:", response)
        
        #TODO: Change to JSON
        self.message = response 
        
        if response is not None and len(response)>1:
            self.message = response
            return 1
        
        self.message = ""
        return 0  
        
        
    def turn_response_into_action(self, response, step = None):
        
        
        #0 -- Extract state and belief from json
        analysis = response.analysis
        observations = [o for o in response.observations] if hasattr(response, "observations") else [response.observation] #initial round only one observation
        hypotheses = [h for h in response.hypotheses]
        
        if self.config["debug"]:
            print(f"DEBUG detail: Agent {self.name} analysis {analysis}.")
            print(f"DEBUG detail: Agent {self.name} observations {observations}.")
            print(f"DEBUG detail: Agent {self.name} hypothesis {hypotheses}.")
        
        # 0 -- possibly add some noise in the hypotheses
        if self.config["parameters"]["noise"] > 0:
            not_last_steps = self.config["max_steps"] - step > 3 if self.config["max_steps"] else True
            if not_last_steps: 
                # affect all hypotheses
                if random.random() < self.config["parameters"]["noise"]:
                    for h in hypotheses: #reset all hypotheses 
                        if self.config["parameters"]["noise_mode"]=="uniform30":
                            h.confidence = random.randint(0, 30)
                        elif self.config["parameters"]["noise_mode"]=="reset":
                            h.confidence = 20
                        elif self.config["parameters"]["noise_mode"]=="shift": #add between -20 and +20
                            h.confidence = h.confidence + (random.randint(0, 40)-20)
                        else:
                            raise ValueError(f"Unknown noise mode {self.config['parameters']['noise_mode']}")
                    if self.config["debug"]:
                        print(f"DEBUG detail: Agent {self.name} change its hypotheses confidence to {[h.confidence for h in hypotheses]}.")
        # 1-- Update observations and beliefs
        self.beliefs = sorted(hypotheses, key=lambda x: x.confidence, reverse=True)[:self.config["parameters"]["max_hypotheses"]]
        self.observations = self.observations + observations
        #print("TP NUM OBSERVATIONS", len(self.observations))
        
        if len(self.observations) > self.config["parameters"]["max_observations"]:
            if self.config["debug"]:
                print(f"DEBUG detail: Agent {self.name} had {self.observations}, keeping only the last {self.config['parameters']['max_observations']}.")
            self.observations = self.observations[:self.config["parameters"]["max_observations"]] #TODO: Synthesize observations !
        
        
        # 2 -- Update state if one hypothesis is above the threshold
        foundBelief = False
        state = None
        if len(self.beliefs)>0 and self.beliefs[0].confidence > self.config["parameters"]["confidence_threshold"]:
            state = self.beliefs[0].digit
            foundBelief = True

        action = bool(state == self.state) #if change consider action ? #NOTE: beware may trigger early stopping
        
        if foundBelief:
            self.state = state
            self.state_confidence = hypotheses[0].confidence
    
                
        return action
        
