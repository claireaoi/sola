
import random
import copy
import json
import os
from src.visualize import generate_gif_from_data_grid, plot_grid, scatter_plot, plot_multiple
from src.utils.utils import json_serial
import time
import yaml
from tqdm import tqdm
import shutil

class GridModel():
    
    def __init__(self, config, id="", output_dir="", with_llm=False, title="", dynamic=True):
        
        self.id = id
        self.config=config
        self.with_llm=with_llm

        self.title=title
        self.output_dir=output_dir

        # Grid size
        self.dimensions = config["grid_size"]  # Now expecting dimensions as a list for n-dimensions

        # Parameters model from config file
        for key, val in config["parameters"].items():
            setattr(self, key, val)

        self._setup_clients()
        
        #initialise population
        self.agents = {}
        self.initialise_population()

        #NOTE: dynamic model, ie agents can move on grid
        self.dynamic=dynamic
        
        self.num_agents = len(self.agents)
 
    def init_anthropic_client(self):
        """
        Initialize the Anthropic client for Claude API access
        """
        from anthropic import Anthropic
        import yaml
        
        # Load Anthropic config from a YAML file
        with open("./config/anthropic.yml") as f:
            anthropic_config = yaml.load(f, Loader=yaml.FullLoader)
        
        client = Anthropic(api_key=anthropic_config["anthropic_api_key"])
        return client
    
    def _setup_clients(self):
        """Set up the appropriate client(s) based on model type.
        When a list of models, setup several clients.
        """
        if isinstance(self.config["llm"], list):
            self.client = {}
            for model_name in self.config["llm"]:
                client = self.get_client(model_name)
                self.client[model_name] = client
        else:
            self.client = self.get_client(self.config["llm"])
       
        
    def get_client(self, model_name):
        """Find appropriate client for the model name.
        """        
        if "llama" in self.config["llm"]: # For local models, no special client setup needed
            client = None
        elif "claude" in model_name:
            client = self.init_anthropic_client()
        elif "gpt" in model_name or "o" in model_name:
            client = self.init_openai_client()
        else:
            raise ValueError("Unsupported model type. Please check the llm parameter in the config.")
        return client

    def init_openai_client(self):
        """
        Init the openai client if needed
        """

        from openai import OpenAI
        with open("./config/openai.yml") as f:
            openai_config = yaml.load(f, Loader=yaml.FullLoader)
        
        client = OpenAI(api_key=openai_config["openai_api_key"])
        return client

    def initialise_population(self):
        # Placeholder method for evaluating a position
        # This method should be implemented based on the specific requirements of the model
        pass

    
    def evaluate_position(self, pos, k=1):
        pass


    def evaluate_empty_positions(self):
        """
        Evaluate the spot, if empty, return None
        """
        #reset
        rated_positions={}

        empty_positions = [(i,j) for i in range(self.dimensions[0]) for j in range(self.dimensions[1]) if (i,j) not in self.agents.keys()]

        for position in empty_positions:
            rated_positions[position]=self.evaluate_position(position, k=self.perception_radius)

        return rated_positions
    
    def get_state_numerical(self, state):
        """
        Return an int index for the state of the agent (needed only for the visualisation)
        #TODO: Should make your own !
        """

        if type(state) == int:
            return state
        
        if type(state) == float:
            return state
        
        if type(state) == str: #NOTE: THAT IS not desirable, code your own
            return 0
        
    
    def step(self, current_step = 0, **kwargs):
        """
        Update the state of all agents in the grid with a certain likelihood.
        #TODO: Refactor this method to improve readability and efficiency.
        # The current implementation uses a deep copy of the agents dictionary and evaluates empty positions in each update step.
        # This can be optimized by avoiding unnecessary deep copies and by caching the evaluated positions.
        """
        count = 0
        num_agents = len(self.agents)
        tp_agents = self.agents if not self.dynamic else {k: copy.copy(agent) for k, agent in self.agents.items()}
        
        possible_positions = None
        
        #0-- rate all empty positions if dynamic
        if self.dynamic: # ie if agents move on the grid
            rated_positions = self.evaluate_empty_positions()
            possible_positions = copy.deepcopy(rated_positions) #TODO better

        # Asynchronous update of agents
        for agent in tp_agents.values():
            r=random.random()
            old_position = copy.deepcopy(agent.position)

            #TODO: If do not update each step keep memory ?
            if (current_step==0) or (r <= self.update_likelihood): #all update initially

                # 1 --- Perception Step
                perception = agent.perceive(tp_agents, step =current_step, **kwargs)
                # 2--- State Update Step 
                if self.dynamic: 
                    action, new_position = agent.update(perception, rated_positions=possible_positions, step=current_step, **kwargs) 
                else:
                    action, new_position = agent.update(perception,  step=current_step, **kwargs)
                count+= bool(action is not None)
                #  --- Update the agent grid if dynamic
                if self.dynamic and (new_position is not None):
                    self.update_agents_grid(agent, old_position, new_position, possible_positions)
                #--- Environment Update Step converting action to environment change possibly
                self.update_environment_from_action(action)
                # 3 -- Tranmission Step: agent may update its message #TODO: only access memory not perception
                agent.transmit(perception, step=current_step)
            else:
                agent.pass_turn()

        return count/num_agents if num_agents>0 else 0
    
    def update_environment_from_action(self, action):
        """
        Update the environment based on the action of the agent (if any other than movement)
        """
        pass
        
        
    def update_agents_grid(self, agent, old_position, new_position,  possible_positions):
        """
        Move agent to new position in grid if dynamic model
        """
        if self.config["debug"]:
            print("DEBUG Moved agent from {} to {} in agent grid".format(old_position, new_position))
        assert new_position==agent.position
        assert new_position in possible_positions.keys()
        del possible_positions[new_position]
        
        # Move agent to new position in gridy
        self.agents[new_position] = agent
        del self.agents[old_position]


    def save_historics(self, output_file):
        """
        Save the historics of the agents (e.g., prompt, messages received, etc.).
        """
        
        historics = {}
        for pos, agent in self.agents.items():
            pos_str = "_".join(map(str, pos))  # Adjust for n-dimensional positions
            historics[pos_str] = agent.trace

        with open(output_file, "w") as f:
            json.dump(historics, f, default=json_serial)


    def evaluate_population(self):
        """
+         Evaluate the whole population according to some specific metrics.
            By default the first key is the one used for the main score tracked.
+         This method is currently a placeholder and should be implemented with the logic to evaluate the population.
+         """
        return {"score": -1}
    
    def wrap_up_run(self, data, num_steps=None):
        """
        Wrap up the run by running additional visualisation or eval
        """
        #TODO Integrate
        
        pass
        
    def early_stopping_condition(self, ratio_actions):
        
        """
        Check if the early stopping condition is met.
        """
        if_early_stopping = all([x <= self.early_stopping_threshold for x in ratio_actions[-self.early_stopping_step:]])
        
        return if_early_stopping

    def run(self, max_steps=None):
        """"
        Run the simulation for max_steps
        Save data every X steps and do some visualisation of the results.

        """

        # For storing agent states at each iteration
        if max_steps is None:
            if self.config["max_steps"]:
                max_steps = self.config["max_steps"]
            else:
                max_steps = 10000 #UPPER BOUND Currently !

        data = {}
        all_metrics = {}  
        ratio_actions=[]
        data[0] = {str(key): val.state for key, val in self.agents.items()}
        initial_metrics = self.evaluate_population()
        for key in initial_metrics.keys():
            all_metrics[key] = [initial_metrics[key]]
    
        main_metric = list(initial_metrics.keys())[0]
        initial_score = initial_metrics.get(main_metric, 0)
        score_population = [initial_score]
    
        if "max_initial_score" in self.config.keys() and self.config["max_initial_score"] is not None and initial_score > self.config["max_initial_score"]:
            print(f"Initial score {score_population} is too high, exiting...")
            #remove folder
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            return None

        # 1-- Run the simulation for max_steps
        for i in tqdm(range(1,max_steps), desc="Running Simulation"):
            
            ratio = self.step(current_step=i)
            
            ratio_actions.append(ratio*100)
            current_metrics = self.evaluate_population()
            for key, value in current_metrics.items():
                all_metrics.setdefault(key, []).append(value)
            current_score = current_metrics.get(main_metric, 0)
            score_population.append(current_score)
            num_updates = int(ratio * self.num_agents)
            tqdm.write(f"Step {i} : {100*ratio} % updates i.e. {num_updates} / {self.num_agents} updates, {main_metric}: {current_score}")

            # Save data every X steps
            if i % self.config["save_every"] == 0:
                data[i] = {str(key): val.state for key, val in self.agents.items()}
            if self.config["visualize_every"] is not None and i% self.config["visualize_every"] == 0:
                data[i] = {str(key): val.state for key, val in self.agents.items()}
                plot_grid(self.config, data[i], title=self.title+f" - Iteration {i}", output_file=self.output_dir+f"grid_{i}.png", with_llm=self.with_llm)
                
            #TODO: Early stopping for the Network model
            if i>self.early_stopping_min_step and self.early_stopping and len(ratio_actions)>self.early_stopping_step:
                if self.early_stopping_condition(ratio_actions):
                    print("Converged, early stopping at {} iterations ଘ(੭*ˊᵕˋ)੭* ̀ˋ since the last update ratio (in %) were {} and threshold at {}".format(i,ratio_actions[-self.early_stopping_step:],self.early_stopping_threshold ))
                    break
                    
        # --Wrap up the run and save the data
        data[i] = {str(key): val.state for key, val in self.agents.items()}
        self.wrap_up_run(data, num_steps = i)
        
        
        # 2--- Plot the final state
        if not self.config["dev"]:

            #create folder if not exist
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            
            #final_score = round(score_population[-1], 3)
            #NOTE: if data string, convert it to float or int for visualisation
            #last_data = {str(key): self.get_state_numerical(val.state) for key, val in self.agents.items()}
                        
            #TODO: color issue in general
            #plot_grid(self.config, last_data, title=self.title+f" - Final Score {final_score}", output_file=self.output_dir+"final_grid.png", with_llm=self.with_llm)
            
            #num actions plot to see evolutions
            scatter_plot(ratio_actions, y_label="Num action",x_label="iterations", output_file=self.output_dir+"ratio_actions.png", every  = self.config["save_every"], max_y=1)

            #score plot to see evolutions
            scatter_plot(score_population, y_label="Score",x_label="iterations", output_file=self.output_dir+"score_pop.png", every = self.config["save_every"], max_y=1)
            
            #Plot other metrics
            plot_multiple(all_metrics, x_label="Iterations", output_file=self.output_dir+"metrics.png", every  = self.config["save_every"])
                
            #Save the score separately as json
            with open(self.output_dir+"score.json", "w") as f:
                s = {"score": score_population}
                json.dump(s, f, default=json_serial)
                
            with open(self.output_dir+"all_metrics.json", "w") as f:
                json.dump(all_metrics, f, default=json_serial)
                
            with open(self.output_dir+"final_metrics.json", "w") as f:
                json.dump(current_metrics, f, default=json_serial)
                
            if len(list(data.keys())) > 0: 
                with open(self.output_dir+"data.json", "w") as f:
                    json.dump(data, f, default=json_serial)
                if self.config["generate_gif"]:
                    generate_gif_from_data_grid(self.config, data_file=self.output_dir+"data.json",output_file=self.output_dir+"grid", title=self.title, with_llm=self.with_llm, score=score_population)
            
        return all_metrics
       
