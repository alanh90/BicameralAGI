# util.py
import json
import os
from dotenv import load_dotenv

load_dotenv()


def get_environment_variable(var_name):
    # Get the environment variable
    var_value = os.getenv(var_name)

    # Check if the variable is set
    if var_value is None:
        raise EnvironmentError(f"{var_name} not found in environment. Ensure the .env file is set up correctly.")

    return var_value


def read_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]


def load_json_file(path):
    with open(path, 'r') as file:
        return json.load(file)
