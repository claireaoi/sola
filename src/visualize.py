


import matplotlib.pyplot as plt
import numpy as np
import json
import imageio
import os
import time
from PIL import Image, ImageDraw, ImageFont, ImageSequence
import math
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from src.utils.utils import pdf
import pandas as pd
from sklearn.linear_model import ElasticNetCV
import seaborn as sns

##########################################
#### VISUALISATION PROCEDURE ####
##########################################


##########################################
#### COLOR UTILS ####
##########################################
#COLORS =["khaki", 'orangered',  "lightsteelblue", 'yellowgreen']
# "khaki", "coral"

# COLORS = [
#     (255, 69, 0),     # 0 – Orange Red
#     (154, 205, 50),   # 1 – Yellow Green
#     (107, 142, 35),   # 2 – Olive Drab
#     (100, 149, 237),  # 3 – Cornflower Blue
#     (218, 112, 214),  # 4 – Orchid
#     (255, 140, 0),   # 5 – Dark orange
#     (160, 82, 45),    # 6 – Sienna 
#     (240, 230, 140),  # 7 – Khaki
#     (176, 196, 222),  # 8 – Light Steel Blue
#     (255, 20, 147)   # 9 – Deep pink
# ]

#(244, 164, 96)    # 9 – Sandy Brown #deep pink

def interpolate_color(color1, color2, factor):
    """Interpolate between two colors."""
    color1_rgb = to_rgb(color1)
    color2_rgb = to_rgb(color2)
    return [(1-factor)*c1 + factor*c2 for c1, c2 in zip(color1_rgb, color2_rgb)]

def state_to_color(state):
    from src.models.classifier.utils import DIGIT_COLORS
    num_colors = len(list(DIGIT_COLORS.values()))
    NORMALIZED_COLORS = [tuple(c / 255.0 for c in color) for color in list(DIGIT_COLORS.values())]

    """Convert state value to an interpolated color."""
    # Convert numpy types to native Python types
    if state is None:
        return "lightgray"
    if isinstance(state, (np.int64, np.int32, np.float64, np.float32)):
        state = state.item()  # Convert to native int or float
    if type(state) == int:
        return NORMALIZED_COLORS[state % num_colors]
    elif type(state) == float:
        if state<0:
            return NORMALIZED_COLORS[0]
        else:
            return NORMALIZED_COLORS[1]
    
    # String state: try converting to int
    if isinstance(state, str):
        try:
            int_state = int(state)
            return NORMALIZED_COLORS[int_state % num_colors]
        except ValueError:
            print(f"[WARNING] Cannot convert string state '{state}' to int — using default color.")
            return "gray"

    print(f"[WARNING] Unrecognized state type: {type(state)} — using fallback color.")
    return "gray"
        


##########################################
#### PLOT GRID  ####
##########################################


def plot_grid(config, data, title="", output_file="", with_llm=False, with_legend=True):

    fig, ax = plt.subplots(figsize=(8, 8))  # You can adjust the figure size as needed

    #NOTE: Because data json has bene stringified because tuple #TODO
    data_grid_tuple = {tuple(map(int, key.strip('()').split(','))): val for key, val in data.items()}

    radius=50 if config["grid_size"][0]>30 else 200

    for key, state in data_grid_tuple.items(): #data is dictionary
        ax.scatter(key[0]+0.5, key[1]+0.5, color=state_to_color(state), s=radius)  # s controls the size of the dots
    
    # A more beautiful title
    ax.set_title(title, fontsize=15,  pad=20) #fontstyle='italic',
    
    ax.set_xlim([0, config["grid_size"][0]])
    ax.set_ylim([0, config["grid_size"][1]])
    ax.set_xticks([])
    ax.set_yticks([])

    # Change spine color and width outter edge
    for spine in ax.spines.values():
        spine.set_edgecolor('lavender')
        spine.set_linewidth(2)  # Adjust for desired thickness
        
    plt.tight_layout()  # To ensure the title and plots don't overlap
    plt.savefig(output_file)
    plt.close()
    
def plot_aggregate(folder, y_label="Seggregation Score",x_label="Steps", every=1):
    data = {}
    #Take all subfolders in that folder
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    for subfolder in subfolders:
        #Take file in that subfolder
        files = os.listdir(subfolder)
        files = [f for f in files if f.endswith("_score.json")][0]
        #Load data from that file json
        with open(subfolder+"/"+files, 'r') as f:
            #name subfolder is name xp
            data[subfolder.split("/")[-1]] = json.load(f)["score"]
    
    #Generate random colors for after
    colors = [interpolate_color("khaki", "coral", i/len(data)) for i in range(len(data))] 
    # Now plot all the data has gathered, each line is an experiment with different color and legend
    plt.figure(figsize=(8,6))
    
    for key, value in data.items():
        plt.plot(value.keys(), value.values(), marker='o', label=key, color = colors.pop(0))
        plt.scatter(value.keys(), value.values(), color='coral', s=60)
    # Adding labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Setting x-ticks to integers from 1 to len(data), jumping by 'every' each time
    tick_positions = range(0, every*len(data), every)
    tick_labels = [str(i+1) for i in tick_positions]  # Generate tick labels. Adjust if you want different labeling
    plt.xticks(tick_positions, tick_labels)

     # Setting y-axis limits
    max_value = max if max is not None else max(data)
    plt.ylim(0, max_value)

    plt.savefig(folder+"/plot_aggregate.png")
    plt.close()
    
                
def scatter_plot(
    data, 
    x=None, 
    title="", 
    y_label="", 
    x_label="", 
    output_file="", 
    legend_labels=None, 
    every=1, 
    max_y=None, 
    with_line=False, 
    line_thickness=1.5,
    regression=False, 
    multiple=False
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    COLORS = [
        "blue", "coral", "yellowgreen", "orchid", 
        "darkkhaki", "lightsteelblue", "orangered", 
        "slategrey", "sandybrown", "gold"
    ]
    if data is None or len(data) == 0:
        print("No data to plot.")
        return

    plt.figure(figsize=(8, 6))
    sns.set_theme()

    if multiple:
        for i, (key, y_vals) in enumerate(data.items()):
            label = legend_labels[key] if legend_labels and key in legend_labels else key
            x_vals = x[key]

            # Zip and sort by x for line plot
            points = sorted(zip(x_vals, y_vals), key=lambda tup: tup[0])
            x_sorted, y_sorted = zip(*points)

            # Plot scatter
            ax = sns.scatterplot(x=x_sorted, y=y_sorted, s=60, color=COLORS[i % len(COLORS)], label=label, alpha=0.9)
            ax.margins(y=0.1) 
            # Optionally add line
            if with_line:
                plt.plot(x_sorted, y_sorted, color=COLORS[i % len(COLORS)], alpha=0.7 , linestyle='-', linewidth=line_thickness)
        
        plt.legend()
    else:
        if x is None:
            x = list(range(0, every * len(data), every))
        sns.scatterplot(x=x, y=data, color='blue', s=60, alpha=0.7)
        if with_line:
            plt.plot(x, data, color='khaki', linestyle='-', linewidth=1.5)
        max_value = max_y if max_y is not None else max(data)
        plt.ylim(0, max_value)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_multiple(metrics_dict, x_label="Iterations", output_file="plots/all_metrics.png", every=1, max_cols=3):
    """
    Plot multiple metrics in a single figure using subplots with a maximum of 3 columns.
    """
    COLORS = ["darkkhaki", 'coral', "lightsteelblue", 'yellowgreen', 'orchid', 'sandybrown']
    num_metrics = len(metrics_dict)
    cols = min(max_cols, num_metrics)
    rows = math.ceil(num_metrics / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if num_metrics > 1 else [axes]
    
    for i, (metric, values) in enumerate(metrics_dict.items()):
       
        axes[i].plot(values, marker='o', linestyle='-',  color=COLORS[i], linewidth=1.5, alpha=0.7)
        axes[i].scatter(range(len(values)), values, color=COLORS[i], s=60)
        
        tick_positions = range(0, every * len(values), every)
        tick_labels = [str(i+1) for i in tick_positions]
        axes[i].set_xticks(tick_positions)
        axes[i].set_xticklabels(tick_labels)
        
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel(metric)
        axes[i].set_title(f"Evolution of {metric}")
        axes[i].legend()
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    
##########################################
#### ANIMATE GRID  ####
##########################################


def generate_gif_from_data_grid(config, data_file="data.json", output_file="", title="", with_llm=False, score=None):

    with open(data_file, 'r') as f:
        data = json.load(f)

    images_path = []

    count=0
    # Generate a plot for each time step
    for key, data_str in data.items():
        title_step=title+"\n t=" + str(key)
        if score is not None and len(score)>count:
            title_step+=" Score: " + str(round(score[count],2))
        plot_grid(config, data_str, title=title_step, output_file=output_file+f"_tp_{key}"+".png", with_llm=with_llm, with_legend=False)
        path=output_file+f"_tp_{key}"+".png"
        images_path.append(path)
        count+=1

    num_extra_iter=math.floor(config["max_steps"]/config["save_every"])-len(list(data.keys())) #because of early stopping
    for _ in range(num_extra_iter):
        images_path.append(path)

    # Generate the GIF
    with imageio.get_writer(output_file+".gif", mode='I', duration=0.6) as writer:
        for image_path in images_path:
            image = imageio.imread(image_path)
            writer.append_data(image)
    
    time.sleep(5)

    # Cleanup temp images
    for img_path in images_path:
        if os.path.exists(img_path):
            os.remove(img_path)

    

##########################################
#### CONCATENATE PROCEDURES ####
##########################################

def concatenate_gifs(gif_paths=None, folder=None, output_path=None, spacing=10, contains=None, title=None):

    if gif_paths is None:
        gif_paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".gif")]
        if contains is not None:
            gif_paths = [path for path in gif_paths if contains in path]

    frames_per_gif = []

    if len(gif_paths) == 0:
        return None
    
    # Extract all frames from each GIF
    for gif_path in gif_paths:
        gif = Image.open(gif_path)
        frames_per_gif.append([frame.copy() for frame in ImageSequence.Iterator(gif)])

    # Determine the GIF with the maximum number of frames
    max_frames = max(len(frames) for frames in frames_per_gif)

    # Ensure all GIFs have the same number of frames
    for frames in frames_per_gif:
        while len(frames) < max_frames:
            frames.append(frames[-1])

    # Assuming all GIFs have the same height
    total_width = sum([frames[0].width for frames in frames_per_gif]) + spacing * (len(gif_paths) - 1)
    height = frames_per_gif[0][0].height

    concatenated_frames = []

    for i in range(max_frames):
        new_frame = Image.new("RGB", (total_width, height), "white")
        x_offset = 0
        for j in range(len(gif_paths)):
            new_frame.paste(frames_per_gif[j][i], (x_offset, 0))
            x_offset += frames_per_gif[j][i].width + spacing

        concatenated_frames.append(new_frame)

    concatenated_frames[0].save(output_path, save_all=True, append_images=concatenated_frames[1:], duration=frames_per_gif[0][0].info['duration'], loop=0)

    return None

def concatenate_pngs(png_paths=None, folder=None, output_path=None, spacing=10, title=None, contains=None):
    
    # Get all png files in the folder
    if png_paths is None:
        png_paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".png")]
        if contains is not None:
            png_paths = [path for path in png_paths if contains in path]

    #print("TP", png_paths[0])
    images = [Image.open(png) for png in png_paths]

    total_width = sum([img.width for img in images]) + spacing * (len(images) - 1)
    height = images[0].height

    if title:
        # Add space for the title at the top
        # font = ImageFont.truetype("arial.ttf", 24)  # Adjust font and size as needed
        text_width, text_height = 1,1#font.getsize(title) #TODO: CHECL
        height += text_height + spacing  # Add space for title + a little more for spacing

    new_image = Image.new("RGB", (total_width, height), "white")

    if title:
        draw = ImageDraw.Draw(new_image)
        draw.text(((total_width - text_width) // 2, spacing), title,  fill="black")#font=font,

    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, height - img.height))
        x_offset += img.width + spacing

    new_image.save(output_path) 


##########################################
#### PLOT SCORE ####
##########################################

def plot_final_score(score, y_label="",x_label="", output_file=""):
    #TODO: ADD VARIANCE
    fig, ax = plt.subplots()
    plt.plot(score.keys(), score.values(), 'ro')
    ax.set_title('Mean Score', fontsize=15) #TODO score name
    ax.set_xlim([0, 1]) #TODO: or depends param measure--
    ax.set_ylim([0, 1.1]) 
    ax.set_xlabel(x_label) #TODO Name parameter make vary
    ax.set_ylabel(y_label)
    if not output_file.endswith(".png"):
        output_file+=".png"
    plt.savefig(output_file)
    plt.close()


##########################################
#### PLOT DISTRIBUTION ####
##########################################
def plot_distribution_hist(data,  mu, std, scale=[1,1], path=None, xlabel="", ylabel='Number of Agents', title=""):
    
    # Plotting
    plt.figure(figsize=(10,6))
    plt.hist(data, bins=np.arange(-3.5, 4.5, 1), align='mid', rwidth=0.7, color='olive', alpha=0.7, label="Population")
    
    # visualize the training data
    bins = np.linspace(-3.5,4.5,100)
    #multiplying to scale approximately
    plt.plot(bins, pdf(bins, mu[0], std[1], scale[0]), color='orange', label="True pdf")
    plt.plot(bins, pdf(bins, mu[1], std[0], scale[1]), color='orange')
    plt.xticks(range(-3, 4))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis='y')

    plt.savefig(path, dpi=300)
    plt.close()
