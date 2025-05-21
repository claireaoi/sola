import random
import time

# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.prompts.persona import NAMES, PERSONAS
import time
import numpy as np
import networkx as nx
from itertools import product
import ollama
import yaml
import json
from itertools import product
from src.utils.utils import fill_placeholders
#from ollama import Llama

##########################################
#### Parent class for LLM Based AGENT ####
##########################################


class LLMAgent:

    def __init__(self, config, state=None, message=None, persona=None, persona_prompt=None,  name="", system_prompt_extra=None, model_name = None, client=None, hidden_state=None):
        """
        Initializes an agent with the given configuration, state, message, persona, and other parameters.
        Args:
            config (dict): Configuration dictionary containing parameters for the agent.
            state (optional): Initial state of the agent. Defaults to None. May be a seed from which to generate the full state.
            hidden_state (optional): Hidden state of the agent. Defaults to None.
            message (optional): Initial message to transmit to neighbors. Defaults to None.
            persona (optional): Persona index to format the system prompt. Defaults to None.
            name (optional): Name of the agent. If None, a random name is chosen from NAMES. Defaults to None.
            system_prompt_extra (List[str], optional): Additional prompt) to append to the system prompt. Defaults to None/
            client (optional): Client instance for the agent. Defaults to None.
        Attributes:
            name (str): Name of the agent.
            state: Current state of the agent.
            config (dict): Configuration dictionary.
            llm_name (str): Name of the language model.
            persona: Persona key or index.
            system_prompt (str): Formatted system prompt based on the persona.
            model_name: Model name 
            llm_model: Initialized language model instance when needed.
            recent_memory (list): List to store recent memory.
            memory (list): List to store memory.
            message: Initial message to transmit to neighbors.
            trace (dict): Dictionary to store prompt, state, and message history.
            META_PROMPTS (dict): Dictionary containing perception, update, and transmit prompts.
            client: Client instance for the agent.
        Prints:
            A message indicating the creation of the agent with its name, persona, and model.
        """

        self.config = config
       
        # System Prompt, state (dynamic state variables) and message
        self.name = random.choice(NAMES) if name is None else name
        if persona: #else may be already defined
            self.persona = persona
            if "personas" in config["parameters"].keys():
                self.persona_prompt = PERSONAS[config["parameters"]["personas"][persona]].format(name=self.name)            
                   
        self.system_prompt = self.get_system_prompt_from_persona(system_prompt_extra)
        self.state = state
        self.hidden_state = hidden_state
        self.message = message if message else self.get_state_as_text()  #TODO or null if no transmission
        
        #LLM
        self.model_name = model_name 
        self.llm_model, self.client = self.initialise_llm(self.model_name, client=client)

        # Memory
        self.recent_memory = []
        self.memory = []
        
        #Historics of state and message if needed
        if self.state:
            self.trace = {"state": [self.state], "message": [self.message], "hidden_state": [self.hidden_state]}
        else:
            self.trace = {"state": [], "message": [], "hidden_state": []}
        # These are meta prompts that are used for the agent's perception, update, and transmission. 
        # They regulate the agent's behavior and interaction with the environment.
        params = config["parameters"]
        init_prompt   = fill_placeholders(config["prompts"]["initialisation"], params) if "initialisation" in config["prompts"].keys() else None
        update_prompt = fill_placeholders(config["prompts"]["update"],    params) if "update" in config["prompts"].keys() else None
        perception_prompt = fill_placeholders(config["prompts"]["perception"], params) if "perception" in config["prompts"].keys() else None
        transmission_prompt = fill_placeholders(config["prompts"]["transmission"], params) if "transmission" in config["prompts"].keys() else None
        self.META_PROMPTS = {"perception": perception_prompt, "update": update_prompt, "transmission": transmission_prompt, "initialisation": init_prompt}

        if config["debug"]:
            print(f"Agent {self.name} created with persona {self.persona} and model {self.model_name}")

    def pass_turn(self):
        """
        Placeholder for the agent's turn logic. This method can be overridden in subclasses.
        """
        #mostly update the trace
        if "state" in self.trace.keys():
            self.trace["state"].append(self.state)
        if "message" in self.trace.keys():
            self.trace["message"].append(self.message)
        if "hidden_state" in self.trace.keys():
            self.trace["hidden_state"].append(self.hidden_state)
        
    def get_persona_as_string(self):
        """
        Return the persona as a string
        """
        return self.config["parameters"]["personas"][self.persona]
    
    def get_system_prompt_from_persona(self, system_prompt_extra):
        """
        Return the system prompt from the persona prompt
        """
        if system_prompt_extra:
            return system_prompt_extra[0].format(name=self.name)+ self.persona_prompt + system_prompt_extra[1].format(name=self.name)
        elif hasattr(self, "persona_prompt") and self.persona_prompt:
            return self.persona_prompt #TODO Change name
        else: #default where nothing
            return "You are a helpful assistant."
    
    def initialise_llm(self, model_name, client = None):
        """
        Initialise the language model and client based on the model name.
        Sometimes the client may have been initialised before (for all the agents) #TODO: make that client initialised above
        """
        llm_model = None
        if "claude" in model_name:
            if client is None:
                from anthropic import Anthropic
                import yaml
                with open("./config/anthropic.yml") as f:
                    anthropic_config = yaml.load(f, Loader=yaml.FullLoader)
                client = Anthropic(api_key=anthropic_config["anthropic_api_key"])
        elif "ollama" in model_name:
            client = None #TODO: llm_model!
        #elif "llama" in model_name:
        #    import llama_cpp
        #    client = None
        #    llm_model = Llama(model_path="./llm/" + model_name + ".bin", n_ctx=self.config["max_tokens"][self.model_name], seed=-1, verbose=False)
        elif "gpt-" in model_name or "o" in model_name:
            if client is None:
                from openai import OpenAI
                with open("./config/openai.yml") as f:
                    openai_config = yaml.load(f, Loader=yaml.FullLoader)
                client = OpenAI(api_key=openai_config["openai_api_key"])
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return llm_model, client

    def ask_llm(self, prompt, num_attempts=1, debug=False, max_tokens=None, response_format=None):
        """
        Ask the language model a question by routing to the appropriate provider
        """
        if debug:
            print(f"Asking message, attempt {num_attempts}: " + prompt)
        
        # Set max_tokens to default if not specified
        if not max_tokens and hasattr(self, "max_tokens"):
            max_tokens = self.max_tokens
        if not max_tokens:
            max_tokens = 10000
        
        # Route to appropriate provider
        if "claude" in self.model_name:
            return self._ask_claude(prompt, max_tokens, response_format)
        elif "ollama" in self.model_name: #TODO not for structured..
            return self._ask_ollama(prompt, max_tokens)
        elif "llama" in self.model_name:#TODO not for structured..
            return self._ask_llama(prompt, max_tokens)
        elif "gpt" in self.model_name:
            return self._ask_openai(prompt, max_tokens, response_format)
        elif "o" in self.model_name:
            return self._ask_openai_o(prompt, max_tokens, response_format)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")
    
    def _ask_ollama(self, prompt, max_tokens):
        """Handle requests to Ollama models"""
        try:
            import ollama
            output = ollama.chat(
                model=self.model_name.split("_")[1] if "_" in self.model_name else "llama2",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options={
                    "num_predict": max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
            return output["message"]["content"]
        except Exception as e:
            print(f"Error in Ollama: {e}")
            return None
    
    def _ask_llama(self, prompt, max_tokens):
        """Handle requests to Llama models"""
        try:
            formatted_prompt = "<<SYS>>\n" + self.system_prompt + "\n<</SYS>>\n\n" + "[INST]" + prompt + "[/INST]"
            output = self.llm_model(
                formatted_prompt, 
                max_tokens=max_tokens, 
                temperature=self.temperature, 
                top_p=self.top_p, 
                repeat_penalty=1.176, 
                top_k=40
            )
            return output["choices"][0]["text"]
        except Exception as e:
            print(f"Error in Llama: {e}")
            return None
    
    def _ask_claude(self, prompt, max_tokens, response_format):
        """
        Handle requests to Claude models using the tools feature to generate structured JSON
        that conforms to Pydantic models.
        """
        try:
            temperature = getattr(self, 'temperature', 0.1)
            top_p = getattr(self, 'top_p', 1.0)
            
            # Create a base request params dict that we'll modify as needed
            request_params = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens if max_tokens is not None else 10000,
                "system": self.system_prompt,
            }

            
            # We'll assume response_format contains Pydantic models as specified
            if response_format is None:
                output = self.client.messages.create(**request_params)
                if output.content and len(output.content) > 0:
                    return output.content[0].text
                else:
                    return None
            else:
                # Extract the Pydantic model schema
                schema = response_format.model_json_schema()
                # Create a tool with the Pydantic schema
                tools = [
                    {
                        "name": "generate_structured_output",
                        "description": f"Generate a structured output according to the provided schema",
                        "input_schema": schema
                    }
                ]
                request_params["tools"] = tools
                request_params["tool_choice"] = {"type": "tool", "name": "generate_structured_output"}
                output = self.client.messages.create(**request_params)
                
                final_output = claude_extract_output_from_tool(output, response_format)
                return final_output
        except Exception as e:
            print(f"Error in Claude API: {e}")
            return None
    
    def _ask_openai(self, prompt, max_tokens, response_format):
        """Handle requests to OpenAI GPT models"""
        try:
            if response_format is None:
                output = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt}, 
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=max_tokens,
                    n=1
                )
                print("output", output.choices[0])
                response = output.choices[0].message.content
                if not response:
                    print("BEWARE output GPT is None", output.choices[0].message)
                    
            else:  # structured response
                output = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt}, 
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=max_tokens,
                    n=1,
                    response_format=response_format
                )
                message = output.choices[0].message
                if (message.refusal):
                    response = None
                    print("Refusal in GPT")
                else:
                    response = message.parsed
            
            return response
            
        except Exception as e:
            print(f"Error in GPT: {e}")
            return None
    
    def _ask_openai_o(self, prompt, max_tokens, response_format):
        """Handle requests to OpenAI 'o' models (specialized variant)"""
        try:
            if response_format is None:
                output = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt}, 
                        {"role": "user", "content": prompt}
                    ],
                    n=1
                )
                print("output", output.choices[0])
                response = output.choices[0].message.content
                if not response:
                    print("BEWARE output GPT is None", output.choices[0].message)
                    
            else:  # structured response
                output = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt}, 
                        {"role": "user", "content": prompt}
                    ],
                    n=1,
                    response_format=response_format
                )
                message = output.choices[0].message
                if (message.refusal):
                    response = None
                    print("Refusal in GPT")
                else:
                    response = message.parsed
            
            return response
            
        except Exception as e:
            print(f"Error in OpenAI 'o' model: {e}")
            return None

    def get_self_perception(self):
        prompts = self.META_PROMPTS["perception"]
        if "self" in prompts.keys():
            return prompts["self"].format(name=self.name, state=self.get_state_as_text())
        else: 
            return ""
        
    def perceive(self, agents, global_perception=None, step =None):
        """
        Generates a perception dictionary for the agent based on its own state, 
        local perceptions from neighboring agents, and an optional global perception.
        Args:
            agents (list): A list of agent objects in the environment.
            global_perception (str, optional): A string representing the global perception 
                               of the environment. Defaults to None.
            step (int, optional): The current step in the simulation. Defaults to None.
        Returns:
            dict: A dictionary containing the agent's perception with keys:
              - "self": The agent's own state.
              - "local": Perceptions from neighboring agents.
              - "global": The global perception of the environment.
                - "aggregated": The aggregated perception of the agent.
                - "reflected": The reflected perception of the agent, reflected with LLM
        """


        prompts = self.META_PROMPTS["perception"]
        perception = {}
        perception["self"] = self.get_self_perception() 
            
        #TODO Add info in message who send it.
        neighbor_messages, neighbors_positions = self.collect_neighbor_messages(agents)
        if neighbor_messages:
            perception["local"] = prompts["local"] + neighbor_messages
        else:
            perception["local"] = ""
            if neighbors_positions:
                perception["local"] +=f"You have {len(neighbors_positions)} neighbors which are situated, compared to you at the following positions: " + "and ".join(neighbors_positions)
            perception["local"] += prompts["local_empty"] if "local_empty" in prompts.keys() else ""
            
        #if self.config["debug"]:
        #    print(f"DEBUG detail: Local perception for agent {self.name}: {perception['local']}")
            
        perception["global"] = ""
        if (global_perception is not None) and "global" in prompts.keys():
            perception["global"] = prompts["global"].format(global_perception=global_perception)

        # Aggregated Perception made from self + Local + Global
        perception["aggregated"] = self.get_context_from_perception(perception)
        
        # Reflect on perception if it is an option
        perception["reflected"] = None
        if self.config["perception_with_reflection"] and "reflection" in self.META_PROMPTS.keys():
            reflection_prompt = self.META_PROMPTS["reflection"].format(name=self.name, perception=perception["aggregated"])
            perception["reflected"] = self.ask_llm(reflection_prompt)
      
        return perception

    def collect_neighbor_positions(self,agents):
        neighbors = self.get_neighbors(agents, k=self.config["parameters"]["perception_radius"])
        neighbors_positions = []

        if len(neighbors) > 0:
            for key, n in neighbors.items():
                neighbors_positions.append("["+key+"]")

        return neighbors_positions
    
    def collect_neighbor_messages(self,agents):
        messages = None
        neighbors = self.get_neighbors(agents, k=self.config["parameters"]["perception_radius"])
        neighbors_positions = []
        all_messages = []

        if len(neighbors) > 0:
            if self.config["perception_with_orientation"] or self.config["perception_with_position"]:
                messages = ""
                for key, n in neighbors.items():
                    neighbors_positions.append("("+key+")")
                    if n.message is not None and n.message != "":
                        if self.config["perception_with_orientation"]:
                            message = f"Message from neighbor at {key}: {n.message}\n"
                        else:
                            message =  f"Message from neighbor at position {key}: {n.message}\n"
                        all_messages.append(message)
            else:
                all_messages = [n.message for n in neighbors if n.message is not None and n.message != ""]
            messages = "\n".join(all_messages)
            #if self.config["debug"]:
            #    print(f"DEBUG detail: Messages from {len(neighbors)} neighbors: {messages}!")
        else:
            if self.config["debug"]:
                print(f"DEBUG detail: No neighbors for agent {self.name}!")
        return messages, neighbors_positions
                
    def update(self, perception, step=None, **kwargs):
        """
        Updates the agent's state and decides whether to transmit a message to its neighbors.
        Args:
            perception (dict): The perception data used to form the context.
            **kwargs: Additional keyword arguments.
        Returns:
            The action which has been taken, if any change, else None.
        """
        if perception is None or perception["aggregated"] is None:
            if self.config["debug"]:
                print(f"DEBUG detail: No perception for agent {self.name}!")
            return None

        #### 1 -- Get Context either aggregated or reflected perception
        context = perception["reflected"] if perception["reflected"] else perception["aggregated"]

        #### 2 -- Update State
        prompt = self.build_update_prompt(context, step = step)
        response = self.ask_llm(prompt)  # , max_tokens=100
        action = self.turn_response_into_action(response, step = step)

        #### 3 -- Update Memory 
        self.update_recent_memory(perception)
        self.update_external_memory(perception)

        #### 4 -- Save Historics
        self.trace["state"].append(self.state)
        self.trace["message"].append(self.message)

        return action
    
    def build_update_prompt(self, context, step = None):
        """
        Build the update prompt
        """
        return context + self.META_PROMPTS["update"].format(name=self.name)
    
    def build_initialization_prompt(self, context):
        """
        Build the initialisation prompt
        """
        if self.META_PROMPTS["initialisation"] is not None:
            return context + self.META_PROMPTS["initialisation"]
        else:
            return None
    
    def turn_response_into_action(self, response, step =None):
        """
        Turn the response of an LLM into an action. This is a placeholder!
        """
        return None
        # if response and "[CHANGE]" in response: #TODO Change this to json 
        #     self.state = self.extract_state_from_text(response.split("[CHANGE]")[1])

    def transmit(self, perception, step=None, **kwargs): #TODO context
        """
        Args:
            context (str): The context in which the message is being transmitted.
            perception (dict): The perception data used to form the context.
            step (int): The current step in the simulation.
            **kwargs: Additional keyword arguments.
        Returns:
            int: Returns 0 if no message is transmitted, otherwise returns 1.
        Behavior:
            - If no transmission prompt is available, do not need to update the message transmitted, it is static.
            - Constructs a prompt based on the current context and previous message.
            - Sends the prompt to the language model to get a response.
            - If the response contains "NONE", clears the message and returns 0.
            - If the response contains "[SHARE]" or "SHARE", updates the message with the content after the keyword and returns 1.
            - If the response does not contain "SHARE", prints an issue message and clears the message.
        """
        
        if (not "transmission" in self.META_PROMPTS.keys()) or self.META_PROMPTS["transmission"] is None:
            return 0
       
        prompt = self.META_PROMPTS["transmission"]
        context = perception["reflected"] if (perception["reflected"] and len(perception["reflected"])>4) else perception["aggregated"]
        
        #NOTE: May give here more information such as previous message transmitted etc #TODO: Function
        prompt = context + prompt.format(name=self.name)

        response = self.ask_llm(prompt,  max_tokens=200)  # max_tokens=100
        if self.config["debug"]:
            #print(f"DEBUG TRANSMIT prompt of {self.name}:", prompt)
            print(f"DEBUG TRANSMIT response of {self.name}:", response)
        
        #TODO: Change to JSON
        self.message = response 
        
        if response is not None and len(response)>1:
            self.message = response
            return 1
        
        self.message = ""
        return 0

    def get_context_from_perception(self, perception):
        """
        Return the context from the perception
        """
        context = ""
        if "self" in perception.keys() and (perception["self"] is not None) and len(perception["self"])>1:
            context += perception["self"]+"\n"
        if "local" in perception.keys():
            context += perception["local"]+"\n"
        if "global" in perception.keys():
            context += perception["global"]+"\n"

        return context

    def get_state_as_text(self):
        """
        Return the textual state of the agent (by default the state itself yet the state could be a seed from which generate full state).
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
        if self.forgetting_rate > 0:
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
        """
        if len(self.recent_memory) >= self.memory_buffer_size:
            self.recent_memory.pop(0)  # Remove the oldest memory
        self.recent_memory.append(memory)  # Add the new memory


##########################################
#### LLM Grid Agent ####
##########################################


class GridLLMAgent(LLMAgent):
    """
    LLM Agent for grid model.
    Attributes:
        position (tuple): The position of the agent in the grid.
        state (dict): The state of the agent.
        message (str): The message associated with the agent.
        persona (str): The persona of the agent.
        system_prompt_extra (str): Additional prompt information.
        client (object): The client associated with the agent.
    Methods:
        __init__(config, position=None, state=None, message=None, persona="", extra_prompt="", client=None):
            Initializes the GridLLMAgent with the given configuration and optional parameters.
        get_neighbors(agents, k=1):
            Retrieves the neighboring agents within a specified distance.
        Initializes the GridLLMAgent with the given configuration and optional parameters.
    """

    def __init__(self, config, position=None, state=None, persona_prompt=None, message=None, name=None,  persona="", model_name = None, system_prompt_extra="", client=None):
        """
        LLM Agent for grid model
        """
        for key, val in config["parameters"].items():
            setattr(self, key, val)
        self.position = tuple(position)  
        LLMAgent.__init__(self, config, state=state, persona=persona, message=message, persona_prompt=persona_prompt, system_prompt_extra=system_prompt_extra,model_name=model_name, client=client, name =name)

    def get_neighbors(self, agents, k=1):
        """Retrieves the neighboring agents within a specified distance.
        Args:
            agents (dict): A dictionary of agents with their positions as keys.
            k (int, optional): The distance within which to search for neighbors. Defaults to 1.
        Returns:
            list or dict: Neighboring agents within the specified distance and orientation if perception_with_orientation is True (dic), else list.
        """

        # Unit direction to name mapping for 2D grid
        DIRECTION_MAP = {
            (0, 1): "Up",
            (0, -1): "Down",
            (-1, 0): "Left",
            (1, 0): "Right",
        }

        def offset_to_name(offset):
            """Convert an offset tuple to a human-readable name."""
            directions = []
            for i, val in enumerate(offset):
                if val != 0:
                    axis_offset = (0, 0)
                    axis_offset = axis_offset[:i] + (1 if val > 0 else -1,) + axis_offset[i+1:]
                    name = DIRECTION_MAP.get(axis_offset)
                    if name:
                        directions.append(f"{name} {abs(val)}")
                    else: # Fallback for diagonals or more dimensions
                        directions.append(f"Axis{i} {'+' if val > 0 else '-'}{abs(val)}")
            return " ".join(directions)

        def offset_to_position(offset):
            """Convert an offset tuple to a position string."""
            position = ",".join(str(self.position[i] + offset[i]) for i in range(len(self.position)))
            return position
        
        offsets = list(product(range(-k, k + 1), repeat=len(self.position)))
        offsets.remove((0,) * len(self.position))  # Exclude self

        if self.config["perception_with_orientation"] or self.config["perception_with_position"]:
            neighbors = {}
        else:
            neighbors = []

        for offset in offsets:
            neighbor_pos = tuple(self.position[i] + offset[i] for i in range(len(self.position)))
            if neighbor_pos in agents:
                if self.config["perception_with_orientation"]:
                    name = offset_to_name(offset)
                    neighbors[name] = agents[neighbor_pos]
                elif self.config["perception_with_position"]:
                    name = offset_to_position(offset)
                    neighbors[name] = agents[neighbor_pos]
                else:
                    neighbors.append(agents[neighbor_pos])

        return neighbors


##########################################
#### LLM NET Agent ####
##########################################


class NetLLMAgent(LLMAgent):
    """
    NetLLMAgent is a specialized LLM (Language Model) Agent designed for grid models. 
    It extends the LLMAgent class and incorporates additional functionality for network-based operations.
    Attributes:
        network (optional): The network structure associated with the agent.
        config (dict): Configuration parameters for the agent.
        state (optional): The initial state of the agent.
        message (optional): The initial message for the agent.
        persona (str): The persona of the agent.
        system_prompt_extra (str): Additional prompt information for the agent.
        client (optional): The client associated with the agent.
    Methods:
        __init__(config, network=None, state=None, message=None, persona="", extra_prompt="", client=None):
            Initializes the NetLLMAgent with the given configuration and optional parameters.
        get_neighbors(network, k=1):
            Retrieves the neighbors of the agent within a specified distance (k) in the network.
    """
    #TODO: add networkx as a requirement and enable more funcytionality therefore
    
    def __init__(self, config, network=None, state=None, message=None, persona_prompt=None,model_name=None, persona="", system_prompt_extra="", client=None, name =None):
        """
        LLM Agent for grid model
        """
        for key, val in config["parameters"].items():
            setattr(self, key, val)

        self.network = network

        LLMAgent.__init__(self, config, state=state, persona=persona, message=message, system_prompt_extra=system_prompt_extra,model_name=model_name, client=client, name =name)

    def get_neighbors(self, network, k=1):
        all_neighbors = nx.single_source_shortest_path_length(network, self.id, cutoff=k)
        all_neighbors.pop(self.id, None)
        return list(all_neighbors.keys())



def claude_extract_output_from_tool(output, response_format):
    """
    Extract structured output from Claude API response and convert it to the specified Pydantic model
    
    Args:
        output: The raw response from Claude API
        response_format: The Pydantic model class to convert the output to
        
    Returns:
        An instance of the specified Pydantic model, or None if extraction fails
    """
    
    # Extract the tool use response
    tool_output = None
    for content in output.content:
        if hasattr(content, 'type') and content.type == "tool_use" and content.name == "generate_structured_output":
            tool_output = content.input
            break
        
    # If we got a tool response, Return an instance of the Pydantic model class
    if tool_output:
        try:
            return response_format.model_validate(tool_output)
        except Exception as e:
            print(f"Failed to validate tool output with model: {e}")
            print("Raw tool output:", tool_output)
    else:
        # Fallback to parsing the text response if no tool output
        if output.content and len(output.content) > 0:
            raw_text = output.content[0].text
            try:
                if '{' in raw_text and '}' in raw_text:
                    json_str = raw_text[raw_text.find('{'):raw_text.rfind('}')+1]
                    parsed_json = json.loads(json_str)
                    try:
                        # Try to convert to Pydantic model
                        return response_format.model_validate(parsed_json)
                    except Exception as e:
                        print(f"Failed to validate parsed JSON with model: {e}")
                        print("Parsed JSON:", parsed_json)
                        return parsed_json
                else:
                    print("Could not find JSON in Claude's response")
                    return None
            except json.JSONDecodeError:
                print("Failed to parse JSON from Claude's response")
                return None
        else:
            print("No content in Claude's response")
            return None