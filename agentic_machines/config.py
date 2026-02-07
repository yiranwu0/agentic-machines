from dotenv import load_dotenv
from pathlib import Path
import os

# Get the directory where this config.py file is located
_config_dir = Path(__file__).parent
_env_path = _config_dir / ".env"

# Load .env file with explicit path
load_dotenv(dotenv_path=_env_path, override=True)

from agentic_machines.utils.llm_utils import LLMCaller

BASE_CONFIG_LIST = [
    # {
    #     'model': "gpt-4o",
    #     "api_key": os.environ.get("OPENAI_API_KEY"),
    #     "api_version": os.environ.get("API_VERSION"),
    #     "azure_endpoint": os.environ.get("AZURE_ENDPOINT"),
    #     "api_type": "azure",
    #     "base_url": os.environ.get("AZURE_ENDPOINT")
    # },
    # {
    #     'model': "gpt-4o-mini",
    #     "api_key": os.environ.get("OPENAI_API_KEY"),
    #     "api_version": os.environ.get("API_VERSION"),
    #     "azure_endpoint": os.environ.get("AZURE_ENDPOINT"),
    #     "api_type": "azure",
    #     "base_url": os.environ.get("AZURE_ENDPOINT")
    # },

    {
        'model': "gpt-4o-mini",
        "api_key": os.environ.get("OPENAI_KEY"),
    },
    {
        'model': "gpt-4o",
        "api_key": os.environ.get("OPENAI_KEY"),
    },
    {
        'model': "o3-mini",
        "api_key": os.environ.get("OPENAI_KEY"),
    },
    {
        'model': "gpt-4.1",
        "api_key": os.environ.get("OPENAI_KEY"),
    },
    {
        'model': "o4-mini",
        "api_key": os.environ.get("OPENAI_KEY"),
        "price": (1.100/1000, 4.4/1000)
    },
    {
        'model': "o3",
        "api_key": os.environ.get("OPENAI_KEY"),
        "price": (2/1000, 8/1000)
    },
    {
        'model': "o3-mini",
        "api_key": os.environ.get("OPENAI_KEY"),
        "price": (1.100/1000, 4.4/1000)
    },
    {
        'model': "computer-use-preview",
        "api_key": os.environ.get("OPENAI_KEY"),
    },
    {
        'model': "gpt-5",
        "api_key": os.environ.get("OPENAI_KEY"),
        "price": (1.25/1000, 10/1000)
    },
    {
        'model': "gpt-5.2",
        "api_key": os.environ.get("OPENAI_KEY"),
        "price": (1.75/1000, 14/1000)
    },
    {
        'model': "gpt-5-mini",
        "api_key": os.environ.get("OPENAI_KEY"),
        "price": (0.25/1000, 2/1000)
    },
    {
        "model": "computer_use_preview",
        "api_key": os.environ.get("OPENAI_KEY"),
        "price": (0.30/1000, 1.2/1000)
    }
]


def set_llm_caller_config(
        cache_seed,
):
    LLMCaller.set_llm_config({"config_list": BASE_CONFIG_LIST, "cache_seed": cache_seed})

def get_llm_config(
        model,
        cache_seed,
        **kwargs
    ):
    filtered = [config for config in BASE_CONFIG_LIST if config["model"] == model]
    if not filtered or len(filtered) == 0:
        raise ValueError(f"Model {model} not found in the config list")
    
    a = {"config_list": filtered, "cache_seed": cache_seed}
    a.update(kwargs)
    return a

