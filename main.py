import datetime
import yaml
import os
from src.visualize import concatenate_pngs, concatenate_gifs, plot_final_score, plot_aggregate,  plot_grid, scatter_plot, plot_multiple
import argparse
import pathlib
import time
import json
from src.utils.utils import load_config_file, generate_random_id, get_num_agents, load_json_data
from evaluate import visualise_experiments_from_folder, add_metrics_to_comparative_metrics
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm

# TODO: Many run for experiment, averaging over the runs
# TODO: Instantiate different relationships between agents

##########################################
#### RUNNING EXPERIMENT ####
##########################################

def run_abm_experiment(config, module):
    #TODO: gather with below
    """
    Run Experiment with ABM
    Make vary the parameter specified in config["main_abm"]
    """
    start_time = time.time()
    id = generate_random_id()
    print(f"RUN {xp.capitalize()} ABM EXPERIMENT {id} \n -  for {config['max_steps']} steps w/ {get_num_agents(config)} agents on grid {config['grid_size']}")
    expe_id = config["name"] + "/abm/" + id
    folder = "outputs/" + expe_id + "/"
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    ModelClass = getattr(module, config["model_class"], None)  # getattr(other_module, class_name, None)
    model = ModelClass(config, id=expe_id)
    score = model.run()

    total_time = round(time.time() - start_time)
    print(f"\n \nExperiment took {total_time}s with a final score of {score}")
    print(f"Experiment saved in {folder}")
    print(f" \n (/¯◡ ‿ ◡)/¯ ~ ┻━┻  Experiment ID {id} Completed (/¯◡ ‿ ◡)/¯ ~ ┻━┻\n" )
    
    return score

def run_one_llm_experiment(config, module, id=None, output_dir=None, num_trials=10):
    
    start_time = time.time()
    
    if num_trials == 0:
        return None
    
    ModelClass = getattr(module, config["model_class"], None)  # getattr(other_module, class_name, None)
    model = ModelClass(config, id=id, output_dir = output_dir)
    metrics = model.run()
    
    if metrics: #if not None else do not save
        model.save_historics(output_dir + "historics.json")
        total_time = round(time.time() - start_time)
        score = metrics["score"][-1] if "score" in metrics else None
        print(f"\n \nExperiment took {total_time} s with a final score of {score}")
        print(f"Experiment saved in {output_dir}")
        with open(output_dir + "config.yml", 'w') as file:
            doc = yaml.dump(config, file)
        print(f"\n  (/¯◡ ‿ ◡)/¯ ~ ┻━┻  Experiment ID {id} Completed (/¯◡ ‿ ◡)/¯ ~ ┻━┻\n" )
    else:
        #Try again
        print(f"\n  (╯°□°）╯︵ ┻━┻  Experiment ID {id} did not complete (╯°□°）╯︵ ┻━┻\n" )
        run_one_llm_experiment(config, module, id=id, output_dir=output_dir, num_trials=num_trials-1)
    return metrics




def run_single_experiment(args):
    config, output_dir, param_dict, run_index, module_name = args
    experiment_modules = {
        "schelling": "src.models.schelling",
      #  "belief": "src.models.belief",
      #  "shape": "src.models.shape",
       # "boid": "src.models.boids",
        "classifier": "src.models.classifier",
    }
        
    module_path = experiment_modules[module_name]
    model_module = __import__(module_path, fromlist=["model"])
    module = model_module.model
        
    # Generate unique ID for experiment
    id = generate_random_id()

    # Create output subdir
    subdir = output_dir + "_".join(f"{k[:3]}{v}" for k, v in param_dict.items()) + f"_{id}/"
    pathlib.Path(subdir).mkdir(parents=True, exist_ok=True)

    # Set parameters in config
    config["parameters"].update(param_dict)

    # Run experiment (with retry logic already inside)
    metrics = run_one_llm_experiment(config, module, id=id, output_dir=subdir)

    return config, metrics, id, param_dict


def run_parallel_llm_experiments(config, output_dir=None, module_name=None):
    comparative_metrics = load_json_data(output_dir, filename="comparative_metrics.json")

    param_names = config["variable_parameters"]
    param_values = config["variable_values"]

    experiment_args = []
    for param_combination in product(*param_values):
        param_dict = dict(zip(param_names, param_combination))
        for run_index in range(config["num_runs"]):
            # Add full config copy for each process
            experiment_args.append((config.copy(), output_dir, param_dict.copy(), run_index, module_name))

            #experiment_args.append((config.copy(), module, output_dir, param_dict.copy(), run_index))

    # Use only physical cores
    with ProcessPoolExecutor(max_workers=config["num_cores"]) as executor:
        futures = {executor.submit(run_single_experiment, args): args for args in experiment_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Running Experiments"):
            cfg, metrics, id, param_dict = future.result()
            if metrics:
                comparative_metrics = add_metrics_to_comparative_metrics(cfg, comparative_metrics, metrics, id, param_dict)

    # Save aggregated results
    with open(output_dir + "comparative_metrics.json", 'w') as file:
        json.dump(comparative_metrics, file)

    with open(output_dir + "config.yml", 'w') as file:
        yaml.dump(config, file)

    visualise_experiments_from_folder(config, comparative_metrics, output_dir=output_dir)

    return comparative_metrics

                
def run_multiple_llm_experiments(config, module, output_dir=None):
    from itertools import product  # Import product for Cartesian product of parameter values

    #Load metrics if exist in this folder
    comparative_metrics = load_json_data(output_dir, filename = "comparative_metrics.json")
        
    # Extract parameter names and values
    param_names = config["variable_parameters"]  # Example: ["polarization", "vacancy"]
    param_values = config["variable_values"]  # Example: [[0.1, 0.5, 0.9], [2, 10, 20, 30]]

    # Iterate over all combinations of variable parameters
    for param_combination in product(*param_values):  # Cartesian product of value lists
        param_dict = dict(zip(param_names, param_combination))  # Map names to values

        for i in range(config["num_runs"]):
            print(f"Running {i}/{config['num_runs']} experiment with {param_dict}")

            # Generate unique ID for experiment
            id = generate_random_id()
            
            # Create output directory using both parameter values
            subdir = output_dir + "_".join(f"{k[:3]}{v}" for k, v in param_dict.items()) + f"_{id}/"
            pathlib.Path(subdir).mkdir(parents=True, exist_ok=True)
            
            # Update config with current parameter combination
            for key, value in param_dict.items():
                config["parameters"][key] = value

            # Run the experiment
            metrics = run_one_llm_experiment(config, module, id=id, output_dir=subdir)
            
            # Add metrics to comparative metrics
            comparative_metrics = add_metrics_to_comparative_metrics(config, comparative_metrics, metrics, id, param_dict)
    
    # Save metrics across runs in json file
    with open(output_dir + "comparative_metrics.json", 'w') as file:
        json.dump(comparative_metrics, file)
    
    #Save config file in main folder too
    with open(output_dir + "config.yml", 'w') as file:
        doc = yaml.dump(config, file)
        
    visualise_experiments_from_folder(config, comparative_metrics, output_dir= output_dir)
        
    return comparative_metrics

       
def run_llm_experiments(config, module, module_name=None):
    """
    Run Experiment for LLM
    Allow for some variable parameters that may record.
    
    """
   
   #model_llm = config["llm"]
    #if "ollama" in model_llm:
    #    model_llm = model_llm.split("_")[1]+" through Ollama"
    
    output_dir = "outputs/" + config["name"] + "/llm/" + config["name_experiment"] + "/"
    # if os.path.exists(output_dir):
    #     print(f"Folder {config['name_experiment']} already exists in {output_dir}. Do you want to continue? (y/n)")
    #     answer = input()
    #     if answer != "y":
    #         print("Exiting")
    #         return None
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if config["variable_parameters"] and config["num_cores"] > 1:
        # Run multiple experiments in parallel
        print(f"Running {xp.capitalize()} COMPARATIVE LLM EXPERIMENTS \n for {config['max_steps']} steps w/ {get_num_agents(config)} agents on grid {config['grid_size']}")
        metrics = run_parallel_llm_experiments(config, output_dir=output_dir, module_name=module_name)
    
    elif config["variable_parameters"] and config["num_cores"] == 1:
        print(f" About to Launch {xp.capitalize()} COMPARATIVE LLM EXPERIMENTS \n for {config['max_steps']} steps w/ {get_num_agents(config)} agents on grid {config['grid_size']}")
        metrics = run_multiple_llm_experiments(config, module, output_dir=output_dir)
    else:
        id = generate_random_id()
        print(f" About to Launch {xp.capitalize()} LLM EXPERIMENT {id} \n for {config['max_steps']} steps w/ {get_num_agents(config)} agents on grid {config['grid_size']}")
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        metrics = run_one_llm_experiment(config, module, id=id, output_dir=output_dir)
        
           
    return metrics



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
    
    parser.add_argument(
        "-c", "--config_file",
        type=str,
        default=None,
        help="The path to the config file to use.",
    )
     
    args = parser.parse_args()
    
    with_llm= bool(args.agent_model=="llm")
    xp = args.experiment  # ✅ now safe to use
    
    if args.config_file is None:
        args.config_file = f"{xp}.yml"

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
      #  "belief": "src.models.belief",
      #  "shape": "src.models.shape",
       # "boid": "src.models.boids",
        "classifier": "src.models.classifier",
    }

    if xp in experiment_modules:
        module_path = experiment_modules[xp]
        model_module = __import__(module_path, fromlist=["model"])
        model = model_module.model
        
        if not with_llm:
            print(f"Running ABM experiment {xp}, version {config['name_experiment']}")
            score = run_abm_experiment(config, module=model)
        else:
            print(f"Running LLM experiment {xp}, version {config['name_experiment']}")
            score = None
            while not score:  # in case of early exiting
                score = run_llm_experiments(config, module=model, module_name = xp)
    else:
        print("~\(≧▽≦)/~")
        raise ValueError(f"Experiment {xp} not implemented yet.")
    


    #AGGREGATE PLOT
    #if score:
    #    plot_aggregate(f"outputs/{xp}/compare")