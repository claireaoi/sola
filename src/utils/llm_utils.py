import tiktoken
import random
import numpy as np


#from openai_secret_manager import assert_credential
# # Ensure you have OpenAI credentials set up to use tiktoken (as of my last update)
# assert_credential() 

#cl100k_base	gpt-4, gpt-3.5-turbo, text-embedding-ada-002
#p50k_base	Codex models, text-davinci-002, text-davinci-003
#r50k_base (or gpt2)	GPT-3 models like davinci

def count_tokens(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens(text, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    
    return num_tokens

# def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
#   """Returns the number of tokens used by a list of messages."""
#   try:
#       encoding = tiktoken.encoding_for_model(model)
#   except KeyError:
#       encoding = tiktoken.get_encoding("cl100k_base")
#   if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
#       num_tokens = 0
#       for message in messages:
#           num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
#           for key, value in message.items():
#               num_tokens += len(encoding.encode(value))
#               if key == "name":  # if there's a name, the role is omitted
#                   num_tokens += -1  # role is always required and always 1 token
#       num_tokens += 2  # every reply is primed with <im_start>assistant
#       return num_tokens
#   else:
#       raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
#   See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


    
