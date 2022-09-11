import os
import yaml


def get_config(config_file):

    if os.path.exists(config_file):
        with open(config_file) as file:
            CONFIG = yaml.load(file, Loader=yaml.FullLoader)
    else:
        print("Not find config.yml file in config/")
    return CONFIG
