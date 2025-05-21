import numpy as np
import matplotlib.pyplot as plt
import os



def get_renormalized_grid(agent_grid, grid_size):
    """
    Renormalizes a 2D grid of agents based on their states.
    Each 2x2 block of the grid is reduced to a single cell in the new grid.
    The new cell's state is determined by the majority state of the block.
    If there is no majority, the state of the bottom-right cell of the block is used.
    If the block is empty, the new cell is also empty.
    """
    
    L = grid_size[0]  # Assuming a square grid
    reduced_L = L // 2  # Reduced grid size after renormalization

    # Initialize new renormalized grid
    renormalized_grid = np.zeros((reduced_L, reduced_L), dtype=int)

    for i in range(reduced_L):
        for j in range(reduced_L):
            # Extract 2x2 block
            block = agent_grid[2*i:2*i+2, 2*j:2*j+2].flatten()
            # Identify bottom-right site
            bottom_right = agent_grid[2*i+1, 2*j+1]
            # Determine majority type in the 2x2 block
            unique, counts = np.unique(block, return_counts=True)
            majority_type = unique[np.argmax(counts)] if len(unique) > 0 else 0
            # Apply renormalization rules
            if majority_type in [1, 2]:  # Majority agent type (e.g., blue or red)
                renormalized_grid[i, j] = majority_type
            elif majority_type == 0:  # Majority vacancies
                renormalized_grid[i, j] = 0
            else:  # No majority, inherit bottom-right site
                renormalized_grid[i, j] = bottom_right

    # Create a dictionary with position as key, and state+1 of the agents...Remove vacancies here
    renormalized_agents = { (i, j): renormalized_grid[i, j] for i in range(reduced_L) for j in range(reduced_L) if renormalized_grid[i, j] != 0 }

    return renormalized_agents
      
def extract_clusters(agent_grid, perception_radius=1):
    """
    Identifies clusters of agents with similar states in a renormalized grid.

    Parameters:
    - agent_grid: 2D numpy array where cells contain integer state values. 0 indicates empty space and are ignored!
    - perception_radius: Neighborhood radius for clustering (default is 1 for direct neighbors).

    Returns:
    - List of clusters, where each cluster is a list of coordinate tuples representing connected agents.
    """
    visited = set()
    clusters = []

    def flood_fill(x, y, state, cluster):
        """Recursive flood-fill algorithm to find connected similar agents."""
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited or agent_grid[cx, cy] != state:
                continue
            visited.add((cx, cy))
            cluster.append((cx, cy))
            # Explore neighbors within the perception radius
            for dx in range(-perception_radius, perception_radius + 1):
                for dy in range(-perception_radius, perception_radius + 1):
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < agent_grid.shape[0] and 0 <= ny < agent_grid.shape[1] and
                            (nx, ny) not in visited and agent_grid[nx, ny] == state):
                        stack.append((nx, ny))

    for i in range(agent_grid.shape[0]):
        for j in range(agent_grid.shape[1]):
            if (i, j) not in visited and agent_grid[i, j] != 0:  # Ignore empty spaces (vacancies)
                cluster = []
                flood_fill(i, j, agent_grid[i, j], cluster)
                if cluster:
                    clusters.append(cluster)

    return clusters

   
def plot_initial_distribution(population, polarization, output_dir):
    """
    Plot the distribution of opinions and degrees in the population.
    """
    
    #TODO: generalise when more than 2 opinions
    # Plot distribution of degrees and opinions
    opinion_degree_product = [degree if opinion==1 else -degree for degree, opinion in population]
    
    # Plot histogram of opinion * degree
    plt.figure(figsize=(8, 6))
    plt.hist(opinion_degree_product, bins=np.arange(-3.5, 4, 1), edgecolor="black", alpha=0.7, color="olive")
    plt.title("Histogram of States * Degree")
    plt.xlabel("State * Degree")
    plt.ylabel("Frequency")

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"distribution_{polarization}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()