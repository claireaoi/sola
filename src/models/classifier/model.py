import itertools
import random
from src.model import GridModel
import numpy as np
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.models.classifier.agent import SelfClassifyingAgent
import os
import numpy as np
import cv2
from scipy.stats import entropy
from src.models.classifier.utils import draw_pattern, generate_pattern, create_legend, animate_grid, plot_evolution_belief, DIGIT_COLORS, load_pattern_json
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, random, textwrap, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from adjustText import adjust_text   # pip install adjustText  (optional but recommended)
from matplotlib.patches import Wedge
from src.models.classifier.utils import get_coordinates

LIGHT_GRAY = (211, 211, 211)          # 0‑255 triplet
LIGHT_GRAY_NORM = tuple(c/255 for c in LIGHT_GRAY)

class SelfClassifyingModel(GridModel):
    def __init__(self, config, output_dir="", id=None, param_name=None):
        title = f"Self Classifying model classes"
        #NOTE: Agent do not move here.
        super().__init__(config, id=id, with_llm=True, title=title, output_dir=output_dir, dynamic=False)

        
    def initialise_population(self):
        """Initialize agents based on the shape pattern."""
        self.count =0 #TODO TEMP
        self.target_shape = self.config["target_shape"]
        self.width = self.config["grid_size"][0]
        self.height = self.config["grid_size"][1]
        self.agents = {}
        
        assert not (self.config["perception_with_position"] and self.config["perception_with_orientation"]), \
        "At most one of 'perception_with_position' or 'perception_with_orientation' can be True"
    
        #if Find file in folder load it as pattern and pattern name
        pattern_files = glob.glob(os.path.join(self.output_dir, "shape_data_*.json"))

        if pattern_files:
            chosen_file = pattern_files[0]
            self.target_shape, self.pattern = load_pattern_json(chosen_file)
            self.config["target_shape"] = self.target_shape
            coordinates = get_coordinates(self.pattern)
        elif self.config["interactive"]:
            self.pattern, self.target_shape, coordinates = draw_pattern(self.config, grid_size=(self.width, self.height), output_dir=self.output_dir)
            self.config["target_shape"] = self.target_shape
        else: #not interactive and no files in the folder
            #try locate in digits folder (in the parent folder of the output_dir) and look for digits/shape_data_<digit>.json
            # actually now saved in /Users/clgl/Github/nca-llm/src/models/classifier/digits
            digits_folder = "./src/models/classifier/digits"
            #digits_folder = os.path.join(os.path.dirname(os.path.dirname(self.output_dir)), "_digits")
            path_pattern = os.path.join(digits_folder, f"shape_data_{self.target_shape}.json")
            pattern_files = glob.glob(os.path.join(digits_folder, f"shape_data_{self.target_shape}.json"))
            print("Trying to load pattern from:", path_pattern)
            if pattern_files:
                chosen_file = pattern_files[0]
                print("Loading pattern from:", chosen_file)
                _, self.pattern = load_pattern_json(chosen_file)
                coordinates = get_coordinates(self.pattern)
            else: #generate from font #TODO: Remove
                raise ValueError(f"Pattern file not found in {self.output_dir} or {digits_folder}.")
                #print("BEWARE No pattern found, generating a new one.")
                #self.pattern = generate_pattern(self.config, self.target_shape, grid_size=(self.width, self.height), output_dir=self.output_dir)
                #coordinates = get_coordinates(self.pattern)
                
                
        for (x,y) in coordinates: #x first here
            model_name = self.config["llm"] if isinstance(self.config["llm"], str) else random.choice(self.config["llm"])
            client = self.client if isinstance(self.config["llm"], str) else self.client[model_name]
            self.agents[(x,y)] = SelfClassifyingAgent(self.config, position=(x, y), model_name=model_name, client=client)
                                           
        print(f"Initialized {len(self.agents)} agents based on the pattern {self.target_shape}.") 
        self.visualize_init_agent_positions()    
        print("Occupied positions:", self.agents.keys())
        if len(self.agents.values()) == 0:
            raise ValueError("No agents were initialized! Check the pattern.")

        for agent in self.agents.values():
            #initialise first hypotheses on their own
            agent.initialisation(self.agents)
        #visualise grid thoughts
        self.visualize_grid_beliefs(step=0)
        self.visualize_grid_all(step=0)
        
        
    def evaluate_population(self):
        """Compute multiple self-organization metrics."""
        #TODO: labels may be ?
        metrics = {}
        converged = self.check_convergence()
        metrics["converged"] = False
        if converged:
            metrics["converged"] = True
            #metrics["convergence_time"] = self.schedule.steps #TODO Track convergence time
        
        self.visualize_grid(step=self.count)
        self.visualize_grid_all(step=self.count)
        self.visualize_grid_beliefs(step=self.count)
        
        labels = [agent.state for agent in self.agents.values() if agent.state is not None]
        if not labels:
            return {"agreement_ratio": 0.0, "entropy": None, "classification_stability": None, "accuracy_ratio": 0.0, "converged": False, "confidence": None}
        
        # Score ratio
        metrics["accuracy_ratio"] = labels.count(self.target_shape) / len(labels)

        # AGreement Ratio
        most_common = max(set(labels), key=labels.count)
        metrics["agreement_ratio"] = labels.count(most_common) / len(self.agents)
        
        # Belief entropy# Compute entropy of classifications 
        _, counts = np.unique(labels, return_counts=True)
        metrics["entropy"] = entropy(counts) if len(counts) > 1 else 0
        
        # Confidence mean
        metrics["confidence"] = np.mean([agent.state_confidence if agent.state_confidence is not None else 0 for agent in self.agents.values()])
        
        # Compute classification stability #TRACK How frequently agents change their minds
        # stability_scores = [len(set(agent.neighborhood_memory)) for agent in self.schedule.agents]
        # metrics["classification_stability"] = np.mean(stability_scores)
        
        self.count = self.count + 1
        
        return metrics
    
    def early_stopping_condition(self, ratio_actions):
        
        """
        #NOTE: here the early stopping condition is based on if the agents have converged in terms of beliefs
        """
        if_early_stopping = self.check_convergence()
        
        return if_early_stopping
    
    
    
    def check_convergence(self):
        """Check if agents converge on the same state with sufficient confidence.

        Conditions:
        - All agents must agree on the same label (no None, and all identical) with confidence > 50% (basic reliability check)
        - At least a ratio of agents (convergence_ratio_agents) must have confidence above convergence_threshold
        
        """
        labels = [agent.state for agent in self.agents.values()]
        if None in labels or len(set(labels)) != 1:
            return False

        confidences = [agent.state_confidence for agent in self.agents.values() if agent.state_confidence is not None]
        if not all(conf > 50 for conf in confidences):
            return False

        threshold = self.config["parameters"]["convergence_threshold"]   # e.g., 80 → 0.8
        ratio_required = self.config["parameters"]["convergence_ratio_agents"]  # e.g., 0.9

        num_above_threshold = sum(conf >= threshold for conf in confidences)
        return (num_above_threshold / len(confidences)) >= ratio_required
    
    def visualize_grid(
        self,
        step: int,
        cell_px: int = 40,        # resolution: pixels *per cell*
        dot_px:  int = 28,        # disc diameter on screen
        title_fs: int = 24,       # title font size
        legend_scale: float = 1.5, # multiplies marker + font in legend
    ):
        # ------------- figure size = cells * pixels -------------
        fig_w = self.width  * cell_px / 100
        fig_h = 1.7 * self.height * cell_px / 100
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

        # ------------- collect scatter data -------------
        xs, ys, colours = [], [], []
        possible_shapes = self.config["shapes"]

        for agent in self.agents.values():
            cx = agent.position[0] + 0.5
            cy = agent.position[1] + 0.5
    
            rgb = (LIGHT_GRAY if agent.state is None else DIGIT_COLORS[agent.state])
            scaled_rgb = tuple(c / 255.0 for c in rgb) 
            alpha = agent.state_confidence / 100 if agent.state_confidence is not None else 1
            colours.append((*scaled_rgb, alpha))  # 0-1 range for matplotlib
            xs.append(cx)
            ys.append(cy)      
            
        # marker size in points^2  (points = pixels * 72 / dpi)
        pt_per_px = 72 / fig.get_dpi()
        s = (dot_px * pt_per_px) ** 2
        ax.scatter(xs, ys, s=s, c=colours)

        # ----------- cosmetics ----------------------------------------------------
        ax.set_xlim(0, self.width);  ax.set_ylim(0, self.height)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("lavender"); spine.set_linewidth(2)
        ax.set_title(f"Step {step}", fontsize=title_fs, pad=25)

        # ----------- legend -------------------------------------------------------
        handles = [mpatches.Patch(color=np.array(DIGIT_COLORS[d])/255, label=str(d))
                for d in possible_shapes]
        ax.legend(handles=handles,
                ncol=max(1, len(handles)//2),
                bbox_to_anchor=(.5, -.08), loc="upper center",
                frameon=False,
                fontsize=title_fs*.6*legend_scale,
                handleheight=1.2*legend_scale,
                handlelength=1.5*legend_scale,
                borderpad=.6*legend_scale,
                markerscale=legend_scale)

        # ----------- save ---------------------------------------------------------
        plt.tight_layout()
        out = os.path.join(self.output_dir, f"grid/step_{step}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight", pad_inches=.05)
        plt.close(fig)
    
    def visualize_init_agent_positions(self, cell_px: int = 40, dot_px: int = 20, title_fs: int = 20):
        fig_w = self.width * cell_px / 100
        fig_h = self.height * cell_px / 100
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

        # Extract agent coordinates
        xs = [pos[0] + 0.5 for pos in self.agents.keys()]  # x is column
        ys = [pos[1] + 0.5 for pos in self.agents.keys()]  # y is row

        # Plot agent dots
        pt_per_px = 72 / fig.get_dpi()
        s = (dot_px * pt_per_px) ** 2
        ax.scatter(xs, ys, s=s, color="black")

        # Grid and axes
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xticks(np.arange(0, self.width + 1))
        ax.set_yticks(np.arange(0, self.height + 1))
        ax.grid(True, which='both', color='lightgray', linewidth=0.5, linestyle='--')
        #ax.invert_yaxis()
        ax.set_title(f"Agent Positions Initially", fontsize=title_fs)

        # Save image
        out = os.path.join(self.output_dir, f"initial_grid_position.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        
        
    def visualize_grid_all(
        self,
        step: int,
        cell_px: int = 40,        # resolution: pixels *per cell*
        dot_px:  int = 28,        # disc diameter on screen
        title_fs: int = 24,       # title font size
        legend_scale: float = 1.5, # multiplies marker + font in legend
    ):
        # ------------- figure size = cells * pixels -------------
        fig_w = self.width  * cell_px / 100
        fig_h = 1.7 * self.height * cell_px / 100
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

        # ------------- collect scatter data -------------
        xs, ys, colours = [], [], []
        possible_shapes = self.config["shapes"]

        for agent in self.agents.values():
            cx = agent.position[0] + 0.5
            cy = agent.position[1] + 0.5
    
            if agent.state is not None and agent.state not in possible_shapes:
                print(f"WARNING Agent state {agent.state} not in shapes {possible_shapes}.")
                        
            if not agent.beliefs:
                # fallback to gray dot if no beliefs
                rgb = LIGHT_GRAY
                scaled_rgb = tuple(c / 255.0 for c in rgb)
                ax.add_patch(plt.Circle((cx, cy), radius=0.3, color=scaled_rgb, alpha=0.4))
                continue

            total_confidence = sum(h.confidence for h in agent.beliefs)
            if total_confidence == 0:
                continue  # or draw gray circle as fallback

            start_angle = 0
            for h in agent.beliefs:
                frac = h.confidence / total_confidence
                end_angle = start_angle + frac * 360
                color_rgb = DIGIT_COLORS.get(h.digit, LIGHT_GRAY)
                color = tuple(c / 255.0 for c in color_rgb)

                wedge = Wedge(center=(cx, cy), r=0.4, theta1=start_angle, theta2=end_angle,
                            facecolor=color, edgecolor="white")
                ax.add_patch(wedge)
                start_angle = end_angle
            
        # marker size in points^2  (points = pixels * 72 / dpi)
        pt_per_px = 72 / fig.get_dpi()
        s = (dot_px * pt_per_px) ** 2
        ax.scatter(xs, ys, s=s, c=colours)

        # ----------- cosmetics ----------------------------------------------------
        ax.set_xlim(0, self.width);  ax.set_ylim(0, self.height)
        ax.set_xticks(np.arange(0, self.width + 1))
        ax.set_yticks(np.arange(0, self.height + 1))
        ax.grid(True, which='both', color='lightgray', linewidth=0.5, linestyle='--')
        #ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("lavender"); spine.set_linewidth(2)
        ax.set_title(f"Step {step}", fontsize=title_fs, pad=25)

        # ----------- legend -------------------------------------------------------
        handles = [mpatches.Patch(color=np.array(DIGIT_COLORS[d])/255, label=str(d))
                for d in possible_shapes]
        ax.legend(handles=handles,
                ncol=max(1, len(handles)//2),
                bbox_to_anchor=(.5, -.08), loc="upper center",
                frameon=False,
                fontsize=title_fs*.6*legend_scale,
                handleheight=1.2*legend_scale,
                handlelength=1.5*legend_scale,
                borderpad=.6*legend_scale,
                markerscale=legend_scale)

        # ----------- save ---------------------------------------------------------
        plt.tight_layout()
        out = os.path.join(self.output_dir, f"grid_allbeliefs/step_{step}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight", pad_inches=.05)
        plt.close(fig)
    
    
    
    def visualize_grid_beliefs(
        self,
        step: int,
        cell_px: int = 40,        # resolution: pixels *per cell*
        dot_px:  int = 28,        # disc diameter on screen
        title_fs: int = 24,       # title font size
        legend_scale: float = 1.5, # multiplies marker + font in legend
        thoughts_prob: float = .30,  # fraction of agents to label
        max_thought_len: int = 80,   # characters before we add “…”
    ):
        # ------------- figure size = cells * pixels -------------
        fig_w = 2*self.width * cell_px / 100
        fig_h = 3*self.height * cell_px / 100
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

        # ------------- collect scatter data -------------
        xs, ys, colours = [], [], []
        possible_shapes = self.config["shapes"]

        for agent in self.agents.values():
            if agent.state is not None and agent.state not in possible_shapes:
                print(f"WARNING Agent state {agent.state} not in shapes {possible_shapes}.")
                
            xs.append(agent.position[0] + 0.5)
            ys.append(agent.position[1] + 0.5)

            rgb = (LIGHT_GRAY if agent.state is None
                else DIGIT_COLORS[agent.state])
            scaled_rgb = tuple(c / 255.0 for c in rgb) 
            alpha = agent.state_confidence / 100 if agent.state_confidence is not None else 1
            colours.append((*scaled_rgb, alpha))  # 0-1 range for matplotlib
            
        # marker size in points^2  (points = pixels * 72 / dpi)
        pt_per_px = 72 / fig.get_dpi()
        s = (dot_px * pt_per_px) ** 2
        ax.scatter(xs, ys, s=s, c=colours)

        # ----------- show thoughts -------------------------------------------
        rng = random.Random(step)          # deterministic per step
        texts = []
        for ag in self.agents.values():
            if ag.beliefs and rng.random() < thoughts_prob:
                x, y = ag.position[0] + .5, ag.position[1] + .5
                belief = str(ag.beliefs[0])
                belief = textwrap.shorten(belief, max_thought_len, placeholder="…")

                # random radial offset so labels don’t all collide
                r, θ = .6 + rng.random()*.4, rng.random()*2*3.1416
                dx, dy = r * np.cos(θ), r * np.sin(θ)
                txt = ax.annotate(
                    f"{ag.name} @ {ag.position}\n"+belief,
                    xy=(x, y), xycoords='data',
                    xytext=(x + dx, y + dy), textcoords='data',
                    fontsize=title_fs * .4,
                    color='dimgray',
                    ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', lw=.6, color='gray', alpha=.7),
                    bbox=dict(boxstyle='round,pad=.2', fc='white', ec='gray',
                            lw=.4, alpha=.8),
                    zorder=3
                )
                texts.append(txt)

        # (optional) push labels apart so none overlap
        if texts:
            adjust_text(texts, ax=ax, expand_text=(1.05, 1.2), only_move={'text':'both'})

        # ----------- cosmetics ----------------------------------------------------
        ax.set_xlim(0, self.width);  ax.set_ylim(0, self.height)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("lavender"); spine.set_linewidth(2)
        ax.set_title(f"Step {step}", fontsize=title_fs, pad=25)

        # ----------- legend -------------------------------------------------------
        handles = [mpatches.Patch(color=np.array(DIGIT_COLORS[d])/255, label=str(d))
                for d in possible_shapes]
        ax.legend(handles=handles,
                ncol=max(1, len(handles)//2),
                bbox_to_anchor=(.5, -.08), loc="upper center",
                frameon=False,
                fontsize=title_fs*.6*legend_scale,
                handleheight=1.2*legend_scale,
                handlelength=1.5*legend_scale,
                borderpad=.6*legend_scale,
                markerscale=legend_scale)

        # ----------- save ---------------------------------------------------------
        plt.tight_layout()
        out = os.path.join(self.output_dir, f"beliefs/step_{step}.png")

        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight", pad_inches=.05)
        plt.close(fig)
    
    
    
    
    def wrap_up_run(self, data, num_steps =0):
        """Save the final grid visualization and metrics."""
        
        print(f"Wrap Up after {num_steps} simulation steps...")
        
        # animate_grid
        animate_grid(self.output_dir)
        
        #Visualise Distribution STate (belief) through time
        plot_evolution_belief(self.config, data, self.output_dir)
        
        #plot_evolution_belief(self.config, data, self.output_dir, weighted=True)
        
        
        




    
