



import datetime
import yaml
import os
from src.visualize import concatenate_pngs, concatenate_gifs, plot_final_score, plot_aggregate,  plot_grid, scatter_plot, plot_multiple
import argparse
import pathlib
import time
import random
import json
from src.utils.utils import load_config_file, generate_random_id, get_num_agents, load_json_data



def add_metrics_to_comparative_metrics(config, comparative_metrics, metrics, id, param_dict):
    """
    Adds the final metric values from an experiment to the comparative metrics dictionary,
    handling multiple variable parameters.

    Parameters:
    - config: dict, experiment configuration settings
    - comparative_metrics: dict, stores comparative results for different parameter values
    - metrics: dict, stores metric results from the current experiment
    - id: str, unique identifier for the experiment
    - param_dict: dict, contains the values of multiple variable parameters

    Returns:
    - Updated comparative_metrics dictionary
    """
    if not metrics:
        return comparative_metrics

    # ifonly one parameter that varies
    
    if len(config["variable_parameters"]) == 1: #NOTE: Was without an s before !!!
        param_name = config["variable_parameters"][0] if isinstance(config["variable_parameters"], list) else config["variable_parameters"]
        for key, value in metrics.items():
            if key not in comparative_metrics:
                comparative_metrics[key] = []
            metric_entry = {
                "id": id,
                "final_metric_value": value[-1],  # Last recorded value of the metric
                "parameter_value": param_dict[param_name],  # Store as a tuple for multi-variable tracking
                "parameter_name": param_name # Store parameter names for clarity
            }
            comparative_metrics[key].append
    else:  
        # Convert parameter dictionary to a hashable tuple for easy comparison
        param_tuple = tuple((k, v) for k, v in param_dict.items())

        for key, value in metrics.items():
            if key not in comparative_metrics:
                comparative_metrics[key] = []        
            metric_entry = {
                "id": id,
                "final_metric_value": value[-1],  # Last recorded value of the metric
                "parameter_value": param_tuple,  # Store as a tuple for multi-variable tracking
                "parameter_name": list(param_dict.keys())  # Store parameter names for clarity
            }
            comparative_metrics[key].append(metric_entry)

    return comparative_metrics

def visualise_experiments_from_folder(config, metrics, output_dir= None):
    
     # Determine variable parameters
    variable_parameters = config["variable_parameters"] if isinstance(config["variable_parameters"], list) else [config["variable_parameters"]]
    
    for name_metric, metric_data in metrics.items():
        
        if len(variable_parameters) == 1:
            # Plot Metrics as function of one variable parameter
            param_name = config["variable_parameters"][0] if isinstance(config["variable_parameters"], list) else config["variable_parameters"]
            x = [data["parameter_value"] for data in metric_data]  
            y = [data["final_metric_value"] for data in metric_data]
            scatter_plot(
                data=y,
                x=x,
                y_label=name_metric,
                x_label=param_name,
                output_file=output_dir + name_metric + ".png",
                multiple=False,
                with_line=True
            )
        
        else:
            # Plot Metrics as function of one variable parameter, with the second parameter being plotted as different colors
            assert len(variable_parameters) == 2, "Only 2 variable parameters are supported for plotting"
            #plot 0 variable as x axis
            plot_metrics_with_two_variable_parameters(name_metric, metric_data, index_x_var=0, index_color_var=1, variable_parameters=variable_parameters,output_dir=output_dir)
            #plot 1 variable as x axis
            plot_metrics_with_two_variable_parameters(name_metric, metric_data, index_x_var=1, index_color_var=0, variable_parameters=variable_parameters,output_dir=output_dir)

def plot_metrics_with_two_variable_parameters(name_metric, metric_data, index_x_var=0, index_color_var=1, variable_parameters=None, output_dir= None):
    
    primary_param = variable_parameters[index_x_var]  # X-axis parameter
    secondary_param = variable_parameters[index_color_var] # Different colors
    print(f"Plotting {name_metric} with {primary_param} on x-axis and {secondary_param} as color.")
    # Organize data for plotting
    #x_values = sorted(set(param[index_x_var][1] for param in [data["parameter_value"] for data in metric_data]))  # Unique x-values
    
    for data in metric_data:
        param = data["parameter_value"]
        if len(param) <= index_color_var:
            print(f"Skipping because param too short: {param}")
            
    # Safely extract the secondary_param values from dict-converted tuples
    all_param_dicts = [dict(data["parameter_value"]) for data in metric_data]
    secondary_values = sorted(set(p[secondary_param] for p in all_param_dicts))
    grouped_data = {val: {"x": [], "y": []} for val in secondary_values}

    #grouped_data = {sec_value: {"x": [], "y": []} for sec_value in sorted(set(param[index_color_var][1] for param in [data["parameter_value"] for data in metric_data]))}

    for data in metric_data:
        param_tuple = dict(data["parameter_value"])
        grouped_data[param_tuple[secondary_param]]["x"].append(param_tuple[primary_param])
        grouped_data[param_tuple[secondary_param]]["y"].append(data["final_metric_value"])

    # Format data for scatter_plot
    formatted_data = {k: v["y"] for k, v in grouped_data.items()}
    formatted_x = {k: v["x"] for k, v in grouped_data.items()}
    legend_labels = {k: f"{secondary_param}={k}" for k in grouped_data.keys()}
    
    # Call scatter_plot function
    scatter_plot(
        data=formatted_data,
        x=formatted_x,
        y_label=name_metric,
        x_label=primary_param,
        output_file=output_dir + name_metric + f"_x{primary_param}.png", #specify what is x
        legend_labels=legend_labels,
        multiple=True,
        with_line=True
    )


def extract_all_params(config, s):
    import re
    param_dic = {}
    
    # Match any group of letters followed by a number (int or float)
    matches = re.findall(r'([a-zA-Z]+)([-+]?\d*\.\d+|\d+)', s)
    
    
    for key, val in matches:
        #identify which variable parameter it is
        key_name = None
        for param in config["variable_parameters"]:
            if key == param[:3]:
                key_name = param
                break
        if not key_name:
            print(f"Warning: Parameter {key} not found in config variabke parameters.",config["variable_parameters"])
            continue
        else:
            # Convert to float or int depending on the value format
            if '.' in val:
                param_dic[key_name] = float(val)
            else:
                param_dic[key_name] = int(val)
    print("Extracted parameters:", param_dic)
    return param_dic

def evaluate_experiment_folder(name, xp):
    
    output_dir = "outputs/" + xp + "/llm/" + name + "/"
    config = load_config_file(config_path  = output_dir+"config.yml")
    
    comparative_metrics = build_comparative_metrics_from_folder(config, output_dir)
    
    # Plot metrics 
    visualise_experiments_from_folder(config, comparative_metrics, output_dir= output_dir)
            

def build_comparative_metrics_from_folder(config, output_dir = None):
    
    
    comparative_metrics = load_json_data(output_dir, filename = "comparative_metrics.json")
    if comparative_metrics:
        metrics_keys = list(comparative_metrics.keys())
        id_present =[m["id"] for m in comparative_metrics[metrics_keys[0]]] #if of present xp
    else:
        id_present = []
    #Add possible missing experiments in folder
    subfolders = [f.path for f in os.scandir(output_dir) if f.is_dir()]
    for subfolder in subfolders:
        filename = subfolder.split("/")[-1]
        bits=filename.split("_")
        #the last part of the name is the id
        id = bits[-1]
        #the other parts are the parameters 
        param_dict = extract_all_params(config, filename)
        if id not in id_present and len(id)>0:
            print(f"Adding missing experiment {id} to comparative metrics")
            metrics = load_json_data(subfolder+"/", filename = "all_metrics.json")
            comparative_metrics = add_metrics_to_comparative_metrics(config, comparative_metrics, metrics, id, param_dict)
    #save comparative metrics back
    with open(output_dir + "comparative_metrics.json", 'w') as file:
        json.dump(comparative_metrics, file)  
        
        
    return comparative_metrics

def evaluate_comparative_experiments(names, xp, evaluation_id =None,  display_names=None):
    """
    Use colors to indicate different experiments (each experiment here is considered one folder)
    """
     
    output_dir = "outputs/" + xp + "/comparative/"
    config = load_config_file(config_path  = "outputs/" + xp + "/llm/" + names[0] + "/config.yml")
    
    comparative_metrics = {}
    if evaluation_id:
        output_dir = output_dir + evaluation_id + "/"
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    for name in names:
        input_dir = "outputs/" + config["name"] + "/llm/" + name + "/"
        metrics = load_json_data(input_dir, filename = "comparative_metrics.json")
        for key, value in metrics.items():
            if key not in comparative_metrics:
                comparative_metrics[key] = {}
            if name not in comparative_metrics[key]:
                comparative_metrics[key][name] = []
            comparative_metrics[key][name].extend(value)
    x, y = {}, {}
    if isinstance(config["variable_parameters"], list):
        assert len(config["variable_parameters"]) == 1, "Currently only one variable parameter is supported for plotting in this version"
    name_variable_parameter = config["variable_parameters"][0] if isinstance(config["variable_parameters"], list) else config["variable_parameters"]
    
    for key, value in comparative_metrics.items():
        x[key], y[key] = {}, {} 
        for name, data_list in value.items():
            x[key][name] = [data["parameter_value"] for data in data_list]
            y[key][name] = [data["final_metric_value"] for data in data_list]
        scatter_plot(y[key], x=x[key], y_label=key.replace("_", " "), x_label=name_variable_parameter, legend_labels =display_names, output_file=output_dir+key+".png", every = 1,  with_line = False, multiple = True)
        

def evaluate_experiments(names, xp, evaluation_id =None,  display_names=None):

    if type(names) == str:
        evaluate_experiment_folder(names, xp)
    else: #list
        evaluate_comparative_experiments(names, xp, evaluation_id, display_names)




if __name__ == "__main__":
  
    
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument(
        "-xp",
        "--experiment",
        type=str,
        choices=["schelling","classifier"], # "shape", "boid", "termites", "belief", 
        required=True,
        help="Name of the experiment to run (schelling or propagation currently implemented).",
    )
    
    
    parser.add_argument(
        "-a",
        "--agent_model",
        type=str,
        choices=["llm", "abm"],
        default="llm",
        help="Agent model: either LLM agent or simple ABM agent.",
    )

    parser.add_argument(
        "-name",
        "--name_experiment",
        nargs="+",  # Accepts multiple values as a list
        default=None,
        help="The name of the experiment(s) to run or evaluate. Provided as a list.",
    )
    
    parser.add_argument(
        "-eval_name",
        "--evaluate_name",
        default="",
        type=str,
        help="To give a name to that comparative evaluation round and the plots.",
    )
    
    xp = parser.parse_args().experiment
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default=f"{xp}.yml",
        help="The name of the config file to use or the experiment ",
    )
    

    args = parser.parse_args()
    with_llm= bool(args.agent_model=="llm")

    # 1-- Load config file
    config = load_config_file(filename = args.config_file)
    if args.name_experiment:
        if len(args.name_experiment)>1:
            config["name_experiment"] = args.name_experiment
        if len(args.name_experiment)==1:
            config["name_experiment"] = args.name_experiment[0]
    if len(args.evaluate_name)>0:
        print("Name evaluate xp", args.evaluate_name)
        evaluation_id = args.evaluate_name
    else:
        evaluation_id = None
    print(f"\nConfig:\n{config}")

    # 3-- Run experiment
    print("\n_____________( ͡°( ͡° ͜ʖ( ͡° ͜ʖ ͡°)ʖ ͡°) ͡°)_____________\n")
    print("_____________________________________________________  \n")
    experiment_modules = {
        "schelling": "src.models.schelling",
     #   "belief": "src.models.belief",
      #  "shape": "src.models.shape",
       # "boid": "src.models.boids",
        "classifier": "src.models.classifier",
    }
    #TODO: add colors here and move to the config file
    display_names ={#NOTE: (Varying Polarization by defaul
        "base30": "Grid 30x30 w/ Biased move",
        "base40": "Grid 40x40 w/ Biased move",
        "optimal40": "Grid 40x40",
        "optimal50Free40": "Grid 50x50, 40% Unoccupied",
        "optimal50Free30": "Grid 50x50",#TODO USually 20 % so could signal it
        "optimal40noise1": "Grid 40x40 + 1% Noise", 
        "optimal40noise5": "Grid 40x40 + 5% Noise",
        "optimal30Free30": "Grid 30x30, 30% Unoccupied",
        "optimal40Free30": "Grid 40x40, 30% Unoccupied",
        "optimal40pol7noise": "Grid 40x40, Polarisation 0.7", #NOTE: (Varying Noise)!
    }
    if xp in experiment_modules:
        module_path = experiment_modules[xp]
        model_module = __import__(module_path, fromlist=["model"])
        model = model_module.model
        
        print(f"Evaluating experiment {xp}, version(s) {config['name_experiment']}")
        score = evaluate_experiments(config["name_experiment"],  xp, evaluation_id=evaluation_id, display_names = display_names)
    else:
        print("~\(≧▽≦)/~")
        raise ValueError(f"Experiment {xp} not implemented yet.")