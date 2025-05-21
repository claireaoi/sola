import matplotlib.pyplot as plt
from src.agentABM import GridABMAgent
import numpy as np
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference



class SchellingABMAgent(GridABMAgent):

    def __init__(self, config, position=None, state=None):
        """
        Note here the state of the Schelling agent is is type .
        """
        super().__init__(config, position=position, state=state)

        for key, val in config["parameters"].items():
            setattr(self, key, val)

    def check_similarity_state(self, state1, state2):
        return state1 == state2

    def perceive(self, agents):
        """
        Return the satisfaction score from its neighbors

        """

        neighbors = self.get_neighbors(agents, k=self.config["parameters"]["perception_radius"])

        # if no neighbors, agent is happy and do not move or random proba ? #TODO
        if len(neighbors) == 0:
            return 0

        # Check ratio of similar types among neighbors if below threshold, unsatisfied
        count_similar = sum([1 if n.state == self.state else 0 for n in neighbors])
        self.score = float(count_similar / len(neighbors))
        unsatisfied = self.score < self.similarity_threshold

        return unsatisfied

    def update(self, perception, rated_positions=None):
        """
        Move the agent to a new position if unsatisfied, based on n-dimensional space.
        """
        if perception == 0:
            return 0, None

        desirable_positions = {k: v for k, v in rated_positions.items() if v[self.state] > self.score}

        # If no desirable positions available
        if not desirable_positions:
            return 0, None

        # Choose a new position based on desirability weights
        weights = [v[self.state] for v in desirable_positions.values()]
        positions = list(desirable_positions.keys())
        new_index = np.random.choice(len(positions), p=np.array(weights) / sum(weights))
        new_position = positions[new_index]

        # Update the agent's position to the new selected position in n-dimensional space
        self.position = new_position

        return 1, new_position
