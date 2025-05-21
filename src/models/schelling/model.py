import itertools
import random
from src.model import GridModel
import numpy as np
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.visualize import plot_distribution_hist, plot_grid
from src.utils.utils import sample_mixture_gaussian
from src.models.schelling.agent import SchellingAgent
from src.models.schelling.agentABM import SchellingABMAgent
import json
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import time
from src.models.schelling.utils import get_renormalized_grid, plot_initial_distribution, extract_clusters


class SchellingLLMModel(GridModel):
    def __init__(self, config, output_dir="", id=None, param_name=None):
        

        if isinstance(config["parameters"]["personas"], list): 
            n_classes = len(config["parameters"]["personas"])
        else:
            n_classes = len(list(config["parameters"]["personas"].keys()))

        assert config["parameters"]["vacancy_percentage"] is not None or config["parameters"]["ratio"] is not None, \
            "Exactly one of 'ratio' or 'vacancy_percentage' must be provided"
        #bool(config["parameters"]["ratio"]) != bool(config["parameters"]["vacancy_percentage"])

        if config["parameters"]["ratio"]:
            assert n_classes == len(config["parameters"]["ratio"]), "Number of classes must be equal to the number of ratios given"
        
        title = f"Schelling model LLM with {n_classes} classes"

        super().__init__(config, id=id, with_llm=True, title=title, output_dir=output_dir, dynamic=True)

    def initialise_population(self):
        """
        A grid or a neighborhood is initialized with a certain number of agents of different types. Some cells are left empty.
        Agents are randomly placed in this grid, with a certain percentage of the grid left unoccupied.
        """
        if self.vacancy_percentage:
            assert self.vacancy_percentage < 100, "Vacancy percentage must be < 100"
            ratio_by_type = (1 - (self.vacancy_percentage/100))/2
            self.ratio = [ratio_by_type, ratio_by_type]
            #print("TP Ratio established as:", self.ratio)
        
        assert sum(self.ratio) <= 1, "Sum of ratio must be <=1"
        
        # create positions in grid
        ranges = [range(dimension) for dimension in self.dimensions]
        self.positions = list(itertools.product(*ranges))
        random.shuffle(self.positions)

        # number positions by types and some slots left empty
        num_agents_by_type = [int(ratio * len(self.positions)) for ratio in self.ratio]
        num_agents = sum(num_agents_by_type)
        self.cached_probabilities = None
        if self.config["use_cached_probability"]:
            cached_probability_path = "src/models/schelling/cache/"+ self.config["cache_id"]+ "/probabilities.json"
            with open(cached_probability_path, "r") as f:
                self.cached_probabilities = json.load(f)
            
        #TODO from bias to ratio
        if self.config["parameters"]["polarization"] is not None:
            degrees, states = self.initialise_distribution_population(num_agents, polarization= self.config["parameters"]["polarization"],bias =self.config["parameters"]["bias"], degrees= self.config["parameters"]["degrees"])
            if self.config["debug"]:
                print("States *Degrees created:", [degree if state==1 else -degree for degree, state in zip(degrees, states)])
        else: #TODO: Check use bias
            degrees = [None for i in range(num_agents)]
            states = [i for i in range(len(self.personas)) for j in range(num_agents_by_type[i])]
            random.shuffle(states)
        
        for n in range(num_agents): 
            model_name = self.config["llm"] if isinstance(self.config["llm"], str) else random.choice(self.config["llm"])
            client = self.client if isinstance(self.config["llm"], str) else self.client[model_name]
            self.agents[self.positions[n]] = SchellingAgent(self.config, state=states[n], degree=degrees[n], position=self.positions[n], cached_probabilities = self.cached_probabilities, model_name=model_name, client=client)
        
        self.evaluate_initial_seggregation()

    def check_similarity_state(self, state1, state2):
        return state1==state2

    def evaluate_position(self, pos, k=1):
        """
        Evaluate position according to ratio of similar types among neighbors
        """
        # TODO: in case more than 2 types, check ratio of similar types among neighbors

        neighbors = [
            self.agents[(x, y)]
            for x in range(pos[0] - k, pos[0] + k + 1)
            for y in range(pos[1] - k, pos[1] + k + 1)
            if (x, y) in self.agents.keys()
        ]
        if len(neighbors) == 0:
            return [1 for _ in range(len(self.personas))]

        similarity_ratios = []
        for i, persona in enumerate(self.personas):
            count_similar = sum([1 for n in neighbors if self.check_similarity_state(n.state, i)])
            similarity_ratios.append(float(count_similar) / len(neighbors))

        return similarity_ratios

    def evaluate_initial_seggregation(self):
        """
        Mean of similarity score, where count the ratio of similar types among neighbors
        """
        similarity = []

        for agent in self.agents.values():
            neighbors = agent.get_neighbors(self.agents, k=self.perception_radius)
            count_similar = sum([1 for n in neighbors if self.check_similarity_state(n.state, agent.state)])
            try:
                similarity.append(float(count_similar) / (len(neighbors)))
            except:
                similarity.append(1)
        self.initial_seggregation = sum(similarity) / len(similarity)



    def initialise_distribution_population(self, n, polarization=0.5, bias=0.0, degrees = [1,2,3]):
        """
        Initialize a population of `n` elements with degrees 1,2,3 and assign each individual an state (0 or 1).

        Parameters:
        - n (int): The total number of elements in the population.
        - polarization (float): A value between 0 and 1 representing the degree of polarization.
        - bias (float): A value between -1 and 1 indicating the bias towards opinion 0 or 1.
                        Negative values favor 0, positive values favor 1.

        Returns:
        - degrees (list): A list of degrees for each individual.
        - states (list): A list of states for each individual.
        
        #NOTE: If only 2 degrees are considered, the polarization is simply the ratio of the higher degree.
        Else, the polarisation create a beta distribution with alpha and beta parameters.
        
        """
        if not (0 <= polarization <= 1):
            raise ValueError("Polarization must be between 0 and 1.")
        if not (-1 <= bias <= 1):
            raise ValueError("Bias must be between -1 and 1.")

        if len(degrees) == 1:
            degrees_distribution = [degrees[0]] * n     
        elif len(degrees) == 2:
            assert degrees[1]>degrees[0], "Degrees must be sorted"
            num_high_degree = int(n * polarization)
            degrees_distribution = [degrees[0]] * (n - num_high_degree) + [degrees[1]] * num_high_degree
        else:
            # Beta distribution - igh polarisation would mean beta near 1
            alpha = 2 #1 + 4 * polarization  #
            beta = 0.01 + 8 * (1 - polarization)  # low polarisation would mean high beta and would mean degrees closer 1 
            degrees_distribution = np.random.beta(a=alpha, b=beta, size=n)
            degrees_distribution = np.digitize(degrees_distribution, bins=[0.33, 0.66])+1  # Map to 1,2,3
        
        print("TP Degrees distribution:", degrees_distribution) 
        # Generate states using bias parameter
        states_distribution = np.random.choice(
            [0, 1], size=n, p=[(1 - bias) / 2, (1 + bias) / 2]
        )

        # Plot Initial Population Distribution
        population_distribution = list(zip(degrees_distribution, states_distribution))
        plot_initial_distribution(population_distribution, polarization, self.output_dir)
        
        return degrees_distribution, states_distribution


      
    def evaluate_population(self):
        """
        Efficiently evaluates multiple self-organization metrics based on self.config["metrics"],
        while optimizing computations to minimize redundant loops and expensive calculations.
        """
        
        metrics = {}

        start_time = time.time()
        # Collect data once to avoid redundant computations
        agent_positions = np.array([agent.position for agent in self.agents.values()])
        agent_states = np.array([agent.state for agent in self.agents.values()])
        
        # Use efficient lookup for neighbors
        all_neighbors = {agent: agent.get_neighbors(self.agents, k=self.perception_radius) for agent in self.agents.values()}

        Ntot = len(self.agents)  # Total number of agents
        
        clusters = self.get_clusters()    
        
        # Precompute mean similarity in a single loop
        if "mean_seggregation" in self.config["metrics"] or "segregation_shift" in self.config["metrics"]:
            similarity_scores = []
            for agent, neighbors in all_neighbors.items():
                count_similar = sum([1 for n in neighbors if self.check_similarity_state(n.state, agent.state)])
                similarity_scores.append(float(count_similar) / len(neighbors) if neighbors else 1)
            mean_seggregation = np.mean(similarity_scores)
            if "mean_seggregation" in self.config["metrics"]:
                metrics["mean_seggregation"] = mean_seggregation
            # Segregation Shift (Avoid recomputation of mean similarity)
            if "segregation_shift" in self.config["metrics"]:
                metrics["segregation_shift"] = mean_seggregation -self.initial_seggregation
            
        # Moran's I (Avoid recalculating distances)
        if "morans_I" in self.config["metrics"]:
            try:
                distances = pairwise_distances(agent_positions)
                W = 1 / (distances + np.eye(len(distances)))  # Inverse distance matrix
                W[np.isinf(W)] = 0
                mean_state = np.mean(agent_states)
                numerator = sum(W[i, j] * (agent_states[i] - mean_state) * (agent_states[j] - mean_state)
                                for i in range(len(agent_states)) for j in range(len(agent_states)))
                denominator = sum((agent_states[i] - mean_state) ** 2 for i in range(len(agent_states)))
                metrics["morans_I"] = (len(agent_states) / sum(sum(W))) * (numerator / denominator)
            except:
                metrics["morans_I"] = None  # Handle edge cases

        # Neighborhood entropy (Compute once)
        if "neighborhood_entropy" in self.config["metrics"]:
            neighbor_entropies = []
            for agent, neighbors in all_neighbors.items():
                neighbor_types = [n.state for n in neighbors]
                _, counts = np.unique(neighbor_types, return_counts=True)
                neighbor_entropy = entropy(counts) if len(counts) > 1 else 0
                neighbor_entropies.append(neighbor_entropy)
            metrics["neighborhood_entropy"] = np.mean(neighbor_entropies)

        # Gini Coefficient (Compute once)
        if "gini_coefficient" in self.config["metrics"]:
            cluster_sizes = np.array([len(c) for c in clusters])
            if len(cluster_sizes) > 1:
                metrics["gini_coefficient"] = np.abs(np.subtract.outer(cluster_sizes, cluster_sizes)).sum() / (
                        2 * len(cluster_sizes) * np.sum(cluster_sizes))
            else:
                metrics["gini_coefficient"] = 0

        if "percolation_seggregation" in self.config["metrics"]:
            metrics["percolation_seggregation"]  = self.percolation_segregation(Ntot, with_renormalization=False, clusters=clusters)
        
        if "percolation_seggregation_renormalized" in self.config["metrics"]:
            metrics["percolation_seggregation_renormalized"]  = self.percolation_segregation(Ntot, with_renormalization=True)

       
        # Fractal Dimension (Compute once)
        if "fractal_dimension" in self.config["metrics"]:
            try:
                nbrs = NearestNeighbors(n_neighbors=2).fit(agent_positions)
                distances, _ = nbrs.kneighbors(agent_positions)
                r = np.log(distances[:, 1])
                log_counts = np.log(np.arange(1, len(r) + 1))
                metrics["fractal_dimension"] = np.polyfit(r, log_counts, 1)[0]  # Slope of log-log plot
            except:
                metrics["fractal_dimension"] = None

        # Save metrics
        self.metrics = metrics
        
        end_time = time.time()
        
        #print(f"TP -- Metrics computed in {end_time - start_time:.2f} seconds")
        
        return metrics
    
    def percolation_segregation(self, n, with_renormalization=False, clusters = None):
        """
        Computes the percolation-based segregation coefficient with or without renormalisation

        Renormalization divides the lattice into 2x2 blocks and assigns a new agent state 
        based on the majority presence in each block.

        Returns:
        - percolation segregation scor ewith or without remormalisation renormalization
        """
        
        if with_renormalization:
            #Want in agent grid to have 1 or 2 since 2 will be used for the vacancy in the get_renormalized_grid
            agent_grid = np.array([[agent.state+1 for agent in row] for row in self.agents.values()])
            renormalized_agent_grid = get_renormalized_grid(agent_grid, self.config["grid_size"])
            n = len(renormalized_agent_grid)  # Total number of agents after renormalization
            clusters = self.extract_clusters(renormalized_agent_grid, perception_radius=self.perception_radius)
        else:
            assert clusters
            
        if n == 0 or len(clusters) == 0:
            return 0  # Avoid division by zero

        cluster_sizes = np.array([len(c) for c in clusters])
        cluster_weights = cluster_sizes / n
        S = np.sum(cluster_sizes * cluster_weights)

        # Normalize segregation score
        percolation_segregation = S * (2 / n)

        return percolation_segregation


    def get_clusters(self):
        """
        Identify clusters of agents with similar states using an iterative flood-fill algorithm.
        Returns a list of clusters, where each cluster is a list of agent IDs.
        """
        visited = set()
        clusters = []

        def flood_fill(start_agent):
            """ Iteratively find all connected similar agents using a stack. """
            stack = [start_agent]
            cluster = []
            
            while stack:
                agent = stack.pop()
                if agent in visited:
                    continue
                visited.add(agent)
                cluster.append(agent)

                for neighbor in agent.get_neighbors(self.agents, k=self.perception_radius):
                    if neighbor not in visited and self.check_similarity_state(agent.state, neighbor.state):
                        stack.append(neighbor)  # Add to stack instead of calling recursively

            return cluster

        for agent in self.agents.values():
            if agent not in visited:
                clusters.append(flood_fill(agent))  # Use iterative flood-fill

        return clusters


    # def get_clusters(self):
    #     """
    #     Identify clusters of agents with similar states using a flood-fill algorithm.
    #     Returns a list of clusters, where each cluster is a list of agent IDs.
        
    #     """
    #     visited = set()
    #     clusters = []


    #     def flood_fill(agent, cluster):
    #         """ Recursively find all connected similar agents. """
    #         if agent in visited:
    #             return
    #         visited.add(agent)
    #         cluster.append(agent)
    #         for neighbor in agent.get_neighbors( self.agents, k=self.perception_radius):
    #             if self.check_similarity_state(agent.state, neighbor.state):
    #                 flood_fill(neighbor, cluster)

    #     for agent in self.agents.values():
    #         if agent not in visited:
    #             cluster = []
    #             flood_fill(agent, cluster)
    #             clusters.append(cluster)

    #     return clusters
    

    def wrap_up_run(self, data, num_steps = None):
        """
        Wrap up the run by running additional visualisation or eval
        """
        #TODO: color issue in general
        print(f"Wrap Up after {num_steps} simulation steps \n-Plotting final configuration...")
        #last_key = list(data.keys())[-1]
        initial_step = list(data.keys())[0]
        final_data = data[num_steps]
        initial_data = data[initial_step]
        plot_grid(self.config, initial_data, title=f"Initial Configuration, Step {initial_step}", output_file=self.output_dir+"initial_grid.png", with_llm=True)
        plot_grid(self.config, final_data, title=f"Final Configuration, Step {num_steps}", output_file=self.output_dir+"final_grid.png", with_llm=True)
        #TODO: animate ?
        #plot_grid(config, data_str, title=title_step, output_file=output_file+f"_tp_{key}"+".png", with_llm=with_llm, with_legend=False)
        #generate_gif_from_data_grid(self.config, data_file=self.output_dir+"data.json",output_file=self.output_dir+"grid", title=self.title, with_llm=self.with_llm, score=score_population)

     

    




class SchellingABMModel(GridModel):

    def __init__(self, config, id="", param_name="similarity_threshold"):

        # ABM model
        param_value = config["parameters"][param_name]
        param_value = str(int(param_value * 100)) + "%"  # TODO change if change param tested
        path = "outputs/{}/p={}_it={}".format(id, param_value, config["max_steps"])
        title = f"Schelling ABM, w/ {param_name}={param_value}."
        super().__init__(config, id=id, with_llm=False, title=title, path=path, dynamic=True)

    def initialise_population(self):
        """
        A grid or a neighborhood is initialized with a certain number of agents of different types. Some cells are left empty.
        Agents are randomly placed in this grid, with a certain percentage of the grid left unoccupied.
        """
        assert sum(self.ratio) <= 1, "Sum of ratio must be <=1"

        # create positions as positions in grid
        ranges = [range(dimension) for dimension in self.dimensions]
        self.positions = list(itertools.product(*ranges))
        random.shuffle(self.positions)

        # number positions by types:
        num_agents_by_type = [int(ratio * len(self.positions)) for ratio in self.ratio]

        count = 0
        # Initialise belieds population
        for i in range(len(self.personas)):
            for j in range(num_agents_by_type[i]):
                self.agents[self.positions[count]] = SchellingABMAgent(self.config, state=i, position=self.positions[count])
                count += 1

    def evaluate_position(self, pos, k=1):
        """
        Evaluate position according to ratio of similar types among neighbors
        #TODO: in case more than 2 types, check ratio of similar types among neighbors

        """
        neighbors = [
            self.agents[(x, y)]
            for x in range(pos[0] - k, pos[0] + k + 1)
            for y in range(pos[1] - k, pos[1] + k + 1)
            if (x, y) in self.agents.keys()
        ]
        if len(neighbors) == 0:
            return [1 for i in range(len(self.personas))]
        count_similar = [sum([1 for n in neighbors if n.state == i]) for i in range(len(self.personas))]
        ratios = [float(count_similar[i]) / len(neighbors) for i in range(len(self.personas))]
        return ratios
    
    def check_similarity_state(self, state1, state2):
        return state1==state2
    
    def evaluate_population(self):
        """
        Mean of similarity score, where count the ratio of similar types among neighbors
        """
        similarity = []
        for agent in self.agents.values():
            neighbors = agent.get_neighbors(self.agents, k=self.perception_radius)
            count_similar = sum([1 for n in neighbors if self.check_similarity_state(n.state, agent.state)])
            try:
                similarity.append(float(count_similar) / (len(neighbors)))
            except:
                similarity.append(1)
        return sum(similarity) / len(similarity)

