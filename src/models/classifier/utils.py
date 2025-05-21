
import numpy as np
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.prompts.persona import NAMES
import numpy as np
import cv2
from PIL import Image
import os
import re
import matplotlib.pyplot as plt
import os, cv2, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#TODO MOVE VISUALISATION and GENERALIS
# Define muted, earthy colors for each digit
DIGIT_COLORS = {
    0: (255, 69, 0),  # Orange Red
    1: (154, 205, 50),  # Yellow Green
    2: (107, 142, 35),  # Olive drab
    3: (100, 149, 237),   # Corn flower blue
    4: (218, 112, 214),  # Orchid
    5: (255, 140, 0),   # 5 – Dark orange
    6: (160, 82, 45),  # Sienna
    7: (240, 230, 140),  # Khaki
    8: (176, 196, 222),  # Light steel blue
    9: (255, 20, 147)   # 9 – Deep pink
}

def plot_evolution_belief(config, data, output_dir="./outputs/self_classifying/llm"):
    """
    Plot the evolution of agents' beliefs over time.
    
    Args:
    - data (list of dicts): List of agent states at each step.
    - output_dir (str): Directory to save the plot.
    """
    # Number of agents
    num_agents = len(data[0])
    num_steps = len(data)
    shapes_present = config["shapes"]          # whatever list/tuple you expect
        
    # Initialize belief count tracking for each digit
    belief_history = {digit: [] for digit in shapes_present}
    
    # Count the beliefs for each step
    for step in range(num_steps):
        state_count = {shape: 0 for shape in shapes_present}
        
        # Count the number of agents believing in each digit
        for state in data[step].values():
            if state is not None:
                assert state in shapes_present, f"State {state} not recognized. Expected one of {shapes_present}."
                state_count[state] += 1
        
        # Calculate percentage and store
        for digit in shapes_present:
            if digit in state_count:
                percentage = (state_count[digit] / num_agents) * 100
            else:
                percentage = 0
            belief_history[digit].append(percentage)
    
    # Plot the evolution of beliefs
    plt.figure(figsize=(12, 8))
    
    for digit, percentages in belief_history.items():
        color_rgb = DIGIT_COLORS.get(digit, (0, 0, 0))
        color = tuple(c / 255.0 for c in color_rgb)  # Normalize for Matplotlib
        plt.plot(range(num_steps), percentages, label=f"Digit {digit}", color=color, linewidth=2)
    
    # Set x-axis ticks to show only integers
    plt.xticks(range(0, num_steps, 1))  
    
    # Add labels and title
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Percentage of Agents Sharing this Belief", fontsize=16)
    plt.title("Evolution of Agents' Beliefs Over Time", fontsize=18)
    plt.legend(title="Digit", loc="upper right", fontsize=16, title_fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the plot
    plot_path = os.path.join(output_dir, "belief_evolution.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Belief evolution plot saved at: {plot_path}")
    
    
def animate_grid(output_dir, gif_name="animated_grid.gif", duration=200):
    """
    Create an animated GIF from a sequence of grid images.
    
    Args:
    - output_dir (str): Directory containing the grid images.
    - gif_name (str): Name of the output GIF file.
    - duration (int): Duration between frames in milliseconds.
    """
    # Get all image file paths
    image_dir = os.path.join(output_dir, "grid")
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    # Sort images by step number using regex to extract numbers
    image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    
    # Load images
    images = [Image.open(os.path.join(image_dir, img)) for img in image_files]
    
    # Save as GIF
    gif_path = os.path.join(output_dir, gif_name)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    print(f"GIF saved at: {gif_path}")
         
# Function to create a legend for the grid visualization with step number
def create_legend(dot_size, digits_present=[0,1,2,3,4,5,6,7,8,9], step=0):
    """Create a legend showing the mapping of colors to digits with the step number."""
    # Adjust legend size to accommodate step number text
    legend_height = dot_size * 4
    legend_width = dot_size * len(digits_present) * 4
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255  # White background
    
    # Add step number at the top
    step_text = f"Step: {step}"
    cv2.putText(legend, step_text, (dot_size, dot_size), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Draw dots and labels for each digit
    for i, digit in enumerate(sorted(digits_present)):
        color_rgb = DIGIT_COLORS.get(digit, (0, 0, 0))
        color = tuple(reversed(color_rgb))  # OpenCV uses BGR
        
        x_pos = i * dot_size * 4 + dot_size
        y_pos = legend_height // 2 + dot_size
        
        # Draw legend dot
        cv2.circle(legend, (x_pos, y_pos), dot_size // 2, color, -1)
        
        # Label the digit
        cv2.putText(legend, str(digit), (x_pos - dot_size // 2, y_pos + dot_size), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return legend

import json
def load_pattern_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["target_shape"], np.array(data["shape_data"])

def generate_pattern(config, pattern, grid_size=(40, 40), output_dir=None):
    
    if pattern in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        return generate_number_pattern(config, grid_size, pattern, output_dir = output_dir)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented.")
    
def save_pattern_json(output_path, target_shape, shape_data):
    data = {
        "target_shape": target_shape,
        "shape_data": shape_data
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
         

# Utility function to generate shape patterns for numbers 0-9
def generate_number_pattern(config, grid_size=(40, 40), digit = 0, output_dir=None):
    """Create binary patterns for numbers 0-9 using OpenCV."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1 # Adjust font scale to fit better
    thickness = 1
    
    #TODO: ADD SOME NOISE ?
    binary_pattern = np.zeros(grid_size, dtype=np.uint8)
    
    # Calculate text size to properly center it
    text_size = cv2.getTextSize(str(digit), font, font_scale, thickness)[0]
    text_x = (grid_size[1] - text_size[0]) // 2
    text_y = (grid_size[0] + text_size[1]) // 2  # Adjust so the digit is vertically centered
    
    # Generate the binary pattern
    cv2.putText(binary_pattern, str(digit), (text_x, text_y), font, font_scale, 255, thickness, cv2.LINE_AA)
    binary_pattern = (binary_pattern > 0).astype(int)  # Convert to binary mask
    
    # Optional: Apply erosion to make the shape thinner
    if config["parameters"]["thin_shape"]:
        kernel = np.ones((2, 2), np.uint8)
        binary_pattern = cv2.erode(binary_pattern.astype(np.uint8), kernel, iterations=1)
    
    
    # Create enlarged visual representation
    dot_size = 10  # Increase this for better visualization
    img_size = (grid_size[0] * dot_size, grid_size[1] * dot_size, 3)
    img = np.ones(img_size, dtype=np.uint8) * 255  # White background
    color_rgb = DIGIT_COLORS.get(digit, (0, 0, 0))  # Get color for the digit
    color = tuple(reversed(color_rgb)) #open cv used bgr
    
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if binary_pattern[x, y] == 1:
                cv2.circle(img, (y * dot_size + dot_size // 2, x * dot_size + dot_size // 2), dot_size // 3, color, -1)  # Draw enlarged dot
    
    if output_dir:
        output_path = os.path.join(output_dir, f"target_shape_{digit}.png")
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, img)
        
    json_path = os.path.join(output_dir, f"shape_data_{digit}.json")
    save_pattern_json(json_path, str(digit), binary_pattern)

    return binary_pattern



def draw_grid_with_pcolormesh(ax, grid):
    ax.clear()
    cmap = ListedColormap(["white", "magenta"])  # background, drawing color
    rows, cols = grid.shape

    # Create mesh grid
    X, Y = np.meshgrid(np.arange(cols+1), np.arange(rows+1))
    ax.set_aspect('equal')
    ax.pcolormesh(X, Y, grid, cmap=cmap, edgecolors='lightgreen', linewidth=1)

    # Hide axes
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    
    

# ------------------------------------------------------------------
# helper – clears previous QuadMesh & draws the new one with borders
# ------------------------------------------------------------------
def draw_grid_with_pcolormesh(ax, grid, cmap):
    # ── remove every previous QuadMesh (works on *all* MPL versions) ──
    for artist in list(ax.collections):      # cast to list → safe iteration
        artist.remove()

    ax.pcolormesh(
        grid,
        cmap=cmap,
        vmin=0, vmax=1,
        edgecolors="lightgreen", linewidth=0.8   # keeps grid lines visible
    )
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)  # Changed to have lower y at bottom #ax.set_ylim(grid.shape[0] - 0.5, -0.5)

# ------------------------------------------------------------------
def draw_pattern(config=None, grid_size=(40, 40), output_dir=None):
    grid     = np.zeros(grid_size, dtype=np.uint8)
    drawing  = {"active": False, "mode": "draw", "last": None}
    # brush size is an **integer side length** (1, 2, 3 …)
    brush = {"size": 1}


    fig, ax  = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    ax.set_frame_on(False)                # ← NEW: remove the axes box
    for spine in ax.spines.values():      # ← NEW: in case a backend re-adds it
        spine.set_visible(False)
    
    cmap = ListedColormap(["white", "magenta"])
    draw_grid_with_pcolormesh(ax, grid, cmap)

    # ---------- helpers ----------
    def set_block(row, col, value):
        s = brush["size"]
        half = (s - 1) // 2
        r0 = max(0, row - half)
        c0 = max(0, col - half)
        r1 = min(grid_size[0] - 1, r0 + s - 1)
        c1 = min(grid_size[1] - 1, c0 + s - 1)
        grid[r0:r1 + 1, c0:c1 + 1] = value
                        
    def interpolate_and_draw(p0, p1, value):
        x0, y0 = p0
        x1, y1 = p1
        n = max(abs(x1 - x0), abs(y1 - y0))

        if n == 0:                 # <-- same cell → just paint it and quit
            set_block(y0, x0, value)
            return

        for i in range(n + 1):
            x = int(round(x0 + (x1 - x0) * i / n))
            y = int(round(y0 + (y1 - y0) * i / n))
            set_block(y, x, value)

    # ---------- UI ----------
    def update_titles():
        fig.suptitle(
            f"Drawing Shape.  Current Mode: {drawing['mode'].capitalize()}, "
            f"Brush Size: {brush['size']}",
            fontsize=14, fontweight="bold"
        )
        ax.set_title(
            "NB:  Toggle 'e' for draw/erase,  and  '+ / -'  to change brush size",
            fontsize=10, pad=12
        )
        fig.canvas.draw_idle()
    # ---------- event callbacks ----------
    def draw_at_cursor(event):
        """Paint/erase only while mouse button held (drawing['active'])."""
        if not drawing["active"] or event.inaxes!=ax \
           or event.xdata is None or event.ydata is None:
            return
        col, row = int(round(event.xdata)), int(round(event.ydata))
        value    = 0 if drawing["mode"] == "erase" else 1
        if drawing["last"] is not None:
            interpolate_and_draw(drawing["last"], (col, row), value)
        else:                               # first point of a stroke / click
            set_block(row, col, value)
        drawing["last"] = (col, row)
        draw_grid_with_pcolormesh(ax, grid, cmap)
        fig.canvas.draw_idle()

    def on_button_press(event):
        drawing["active"] = True
        drawing["last"]   = None
        draw_at_cursor(event)               # executes the first point

    def on_button_release(event):
        drawing["active"] = False
        drawing["last"]   = None

    def on_key(event):
        if event.key == 'e':
            drawing["mode"] = "erase" if drawing["mode"] == "draw" else "draw"
        elif event.key in ['+', '=']:
            brush["size"] = min(15, brush["size"] + 1)
        elif event.key in ['-', '_']:
            brush["size"] = max(1,  brush["size"] - 1)
        elif event.key in ['r', 'R']:
            grid[:, :] = 0
        update_titles()
        draw_grid_with_pcolormesh(ax, grid, cmap)

    # ---------- wire up ----------
    update_titles()
    fig.canvas.mpl_connect('button_press_event',  on_button_press)
    fig.canvas.mpl_connect('motion_notify_event', draw_at_cursor)
    fig.canvas.mpl_connect('button_release_event',on_button_release)
    fig.canvas.mpl_connect('key_press_event',     on_key)

    plt.show()

    # --------- everything below unchanged (save, JSON, etc.) ---------
    while True:                                   # keep asking until input is valid
        target_shape = input("What digit did you draw (0-9)? ").strip()
        if target_shape in "0123456789":          # exit when user typed 0-9
            break
    digit = int(target_shape)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        dot = 10
        img = np.ones((grid_size[0]*dot, grid_size[1]*dot, 3), dtype=np.uint8)*255
        color = DIGIT_COLORS.get(digit, (0,0,0))
        for r in range(grid_size[0]):
            for c in range(grid_size[1]):
                if grid[r,c]:
                    cv2.circle(img, (c*dot+dot//2, r*dot+dot//2),
                               dot//3, tuple(reversed(color)), -1)
        cv2.imwrite(os.path.join(output_dir, f"target_shape_{digit}.png"), img)
        # Only flip the grid for the JSON file
        # This creates a new array where the top of the canvas (0) becomes the highest y-value
        flipped_grid = np.flipud(grid).copy()
        
        coordinates = get_coordinates(flipped_grid)
        
        # Convert the numpy array to a list for JSON serialization
        # This way the coordinates in the JSON will have (0,0) at bottom-left
        # and y increasing upward
        grid_list = flipped_grid.tolist()
        
        save_pattern_json(os.path.join(output_dir,
                                      f"shape_data_{digit}.json"), digit, grid_list)


    return flipped_grid, digit, coordinates


def get_coordinates(grid):
    """
    Print all non-zero coordinates in a grid in (x, y) format.
    
    Args:
        flipped_grid: NumPy array containing the grid data
    """
    print("\nNon-zero coordinates (x, y format):")
    coordinates = []
    
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] != 0:
                coordinates.append((x, y))
    
    # Sort by y first (descending) then by x (ascending) to make it easier to visualize
    coordinates.sort(key=lambda coord: (-coord[1], coord[0]))
    print("Agents coordinates in the grid (HERE x FIRST!):", coordinates)
    
    return coordinates
    
    