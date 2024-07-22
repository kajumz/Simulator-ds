import os
import yaml


def yaml_to_env(config_text: str) -> str:
    config = yaml.safe_load(config_text)
    env_list = []
    traverse_config(config, '', env_list)
    return '\n'.join(env_list)


def traverse_config(config, prefix, env_list):
    for key, value in config.items():
        if isinstance(value, dict):
            traverse_config(value, f'{prefix}{key}.', env_list)
        else:
            env_list.append(f'{prefix}{key}={value}')


def env_to_yaml(env_text: str) -> str:
    config = {}
    for line in env_text.split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = convert_value(value)

    yaml_data = yaml.dump(config, default_flow_style=False)
    return yaml_data

def convert_value(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value