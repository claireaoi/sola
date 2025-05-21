

###### PERCEIVE ########
PERCEPTION={
    "local": """Here are the persona you have recently interacted with: {local_perception}.""",
    "local_empty": """You are currently living relatively isolated: you have not interacted with anyone.""",
    "local_neighbors": """{name}, which shared the following: "{message}" .""",
    "self": """Here is your current beliefs: {state}.""",
    "global": """Here is the recent global news you have been exposed to: {global_perception}. """
}

###### UPDATE ########
UPDATE="""Reflect upon this context, to see if and how {name} has evolved its beliefs.
You ({name}) can decide either to update your beliefs, or to keep your beliefs. Your response should include:
(1) An analysis of the context and cognitive mechanisms at play in {name} head recently. 
(2) The decision: answer "[KEEP]" if {name} has not modified its beliefs, or "[CHANGE]" followed by the new set of beliefs {name} behold.
"""
###### TRANSMIT ########
TRANSMIT="""Reflect upon this context, and decide if {name} should share something to its neighbors.
You can either decide not to share anything by answering [NONE] or to share something by answering [SHARE] followed by the message you want to share with your community.
Do not repeat yourself.
"""
#TODO: test "try not to repeat yourself ?""

TRANSMIT_spe="""Reflect upon this context, and decide what {name} should share with {name2}. 
You can either decide not to share anything by answering [NONE] or to share something by answering [SHARE] followed by the message you want to share with {name}2.
"""


META_PROMPTS={
    "perception": PERCEPTION,
    "update": UPDATE,
    "transmission": TRANSMIT
}

