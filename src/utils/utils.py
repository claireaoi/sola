
import numpy as np

########################################
#### DISTRIBUTIONS ########################
########################################

def sample_mixture_gaussian(bias, means=[-1.5, 1.5], std=[1.0, 1.0],  n_samples = 1000):
    """
    bias: between 0 and 1, weight over first distribution
    polarisation: between 0 and 1, how much the distribution are accentuated

    standard deviation : w/ sqrt
    scale variance is max variance
    """
    n1=int(n_samples*(1-bias))
    n2=int(n_samples*bias)

    x1 = np.random.normal(means[0], std[0], n1)
    x2 = np.random.normal(means[1], std[1], n2)

    X = np.array(list(x1) + list(x2))
    np.random.shuffle(X)
    print("Dataset shape:", X.shape)

    return X


def pdf(data, mean: float, variance: float, num_agents: int):
  # A normal continuous random variable.
  s1 = 1/(np.sqrt(2*np.pi*variance))
  s2 = num_agents * np.exp(-(np.square(data - mean)/(2*variance)))
  #scale it by num agents
  return s1 * s2

 
####################################
#### MIX ########################
####################################

import re
from typing import Mapping, Any

_placeholder = re.compile(r"#([A-Za-z_]\w*)")          # → #max_hypotheses

def _substitute(text: str, params: Mapping[str, Any]) -> str:
    """Replace #placeholders in a single string."""
    def repl(match):
        key = match.group(1)
        try:
            return str(params[key])
        except KeyError as exc:
            raise KeyError(f"Parameter '{key}' not found in params") from exc
    return _placeholder.sub(repl, text)


def fill_placeholders(data, params: Mapping[str, Any]) -> Any:
    """
    Recursively walk *data* and replace every #name with params['name'].

    Works on str / dict / list / tuple / set (including nested mixes).
    """
    if data is None:
        return None

    # ─── case 1: string ─────────────────────────────────────────────────────────
    if isinstance(data, str):
        return _substitute(data, params)

    # ─── case 2: mapping (dict, defaultdict, OrderedDict …) ─────────────────────
    if isinstance(data, Mapping):
        return data.__class__(                          # keep original mapping type
            {k: fill_placeholders(v, params) for k, v in data.items()}
        )

    # ─── anything else is returned untouched ────────────────────────────────────
    return data

import random
def generate_random_id():
    # File paths
    adjective_file = 'src/utils/data/A.txt'
    noun_file = 'src/utils/data/N.txt'

    # Load adjectives and nouns from the files
    with open(adjective_file, 'r') as a_file:
        adjectives_from_file = a_file.read().splitlines()

    with open(noun_file, 'r') as n_file:
        nouns_from_file = n_file.read().splitlines()

    # Randomly pick one adjective and one noun
    a, n = "", ""
    while len(a)==0 or len(a)>8:
        a = random.choice(adjectives_from_file)
    while n == "" or len(n)>8:
        n = random.choice(nouns_from_file)
    id = a + "-"+ n
    return id

def get_num_agents(config):
    #TODO for grid more dim
    num_spots= config["grid_size"][0]*config["grid_size"][1]
    
    if "ratio" in config["parameters"] and config["parameters"]["ratio"]:
        ratio_agents= sum(config["parameters"]["ratio"])
        return int(num_spots*ratio_agents)
    elif "vacancy_percentage" in config["parameters"] and config["parameters"]["vacancy_percentage"]:
        return int(num_spots*(100-config["parameters"]["vacancy_percentage"]))
    
    return None 


####################################
#### SAVE ########################
####################################

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, 
                          np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")



####################################
#### LOAD ########################
####################################'
import os
import yaml
import json
def load_config_file(filename =None, config_path =None ):
    if filename:
        config_path = os.path.join(os.getcwd(), "config/" + filename)
    else:
        assert config_path, "No config file provided"

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def load_json_data(output_dir, filename = "comparative_metrics.json"):
    if os.path.exists(output_dir + filename):
        with open(output_dir + filename, 'r') as file:
            data = json.load(file)
    else:
        data = {}
    return data   