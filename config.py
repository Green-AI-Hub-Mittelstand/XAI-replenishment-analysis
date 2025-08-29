import json
import os
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config():
    """
    Loads configuration from config.json or falls back to config.sample.json.

    This function implements the pattern suggested by your supervisor:
    - It first tries to load the user-specific `config.json`.
    - If `config.json` is not found, it logs a warning and loads the
      default `config.sample.json`.
    - This ensures the app can always run, even without a custom config.
    """
    config_path = 'configs/config.json'
    sample_config_path = 'configs/config.sample.json'

    # Check for the primary config file
    if os.path.exists(config_path):
        logging.info(f"Loading configuration from '{config_path}'")
        with open(config_path, 'r') as f:
            return json.load(f)
    # Fallback to the sample config file
    elif os.path.exists(sample_config_path):
        logging.warning(f"'{config_path}' not found. Falling back to '{sample_config_path}'.")
        with open(sample_config_path, 'r') as f:
            return json.load(f)
    # If neither exists, raise an error
    else:
        raise FileNotFoundError(
            f"Configuration file not found. Please create '{config_path}' or ensure "
            f"'{sample_config_path}' exists in the project root."
        )


# Load the settings once when the module is imported
settings = load_config()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))