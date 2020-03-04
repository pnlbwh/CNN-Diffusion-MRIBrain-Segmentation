import ast
import configparser
from collections.abc import Mapping
from os import getenv


def parse_values(config):
    config_parsed = {}
    for section in config.sections():
        config_parsed[section] = {}
        for key, value in config[section].items():
            config_parsed[section][key] = ast.literal_eval(value)
    return config_parsed


class Settings(Mapping):
    def __init__(self, setting_file=getenv('COMPNET_CONFIG')):

        if not setting_file:
            raise EnvironmentError('environment variable COMPNET_CONFIG is not defined, see docs/README.md for details')

        config = configparser.ConfigParser()
        config.read(setting_file)
        self.settings_dict = parse_values(config)

    def __getitem__(self, key):
        return self.settings_dict[key]

    def __len__(self):
        return len(self.settings_dict)

    def __iter__(self):
        return self.settings_dict.items()
    