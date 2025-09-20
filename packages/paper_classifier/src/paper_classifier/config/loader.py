"""
Configuration loader for YAML-based configs with inheritance and CLI override support.
"""
import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import re


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in config values."""
    if isinstance(value, str):
        # Pattern to match ${VAR} or ${VAR:-default}
        pattern = r'\$\{([^}]+)\}'
        
        def replacer(match):
            var_expr = match.group(1)
            if ':-' in var_expr:
                var_name, default_val = var_expr.split(':-', 1)
                return os.environ.get(var_name, default_val)
            else:
                return os.environ.get(var_expr, match.group(0))
        
        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    else:
        return value


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file with support for 'extends' directive."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance via 'extends'
    if 'extends' in config:
        base_config_name = config.pop('extends')
        base_config_path = config_path.parent / base_config_name
        base_config = load_yaml_config(str(base_config_path))
        
        # Deep merge base config with current config
        merged_config = deep_merge(base_config, config)
        return merged_config
    
    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_config(config: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten nested config dictionary for easier CLI argument mapping."""
    items = []
    
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_config(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def unflatten_config(flat_config: Dict[str, Any], sep: str = '_') -> Dict[str, Any]:
    """Unflatten a flat dictionary back to nested structure."""
    result = {}
    
    for flat_key, value in flat_config.items():
        keys = flat_key.split(sep)
        current = result
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # If the current value is not a dict, skip this path
                break
            current = current[key]
        else:
            # Only set the value if we successfully traversed all keys
            if isinstance(current, dict):
                current[keys[-1]] = value
    
    return result


class ConfigArgumentParser:
    """Parser that combines YAML config with CLI arguments."""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.config_args = {}
        
    def add_config_argument(self):
        """Add the --config argument."""
        self.parser.add_argument(
            '--config',
            type=str,
            help='Path to YAML configuration file'
        )
        
    def add_arguments_from_config(self, config: Dict[str, Any]):
        """Dynamically add arguments based on config structure."""
        flat_config = flatten_config(config)
        
        for key, value in flat_config.items():
            arg_name = f'--{key.replace("_", "-")}'
            
            # Determine argument type based on value
            if isinstance(value, bool):
                self.parser.add_argument(
                    arg_name,
                    type=lambda x: x.lower() == 'true',
                    default=None,
                    help=f'Override for {key}'
                )
            elif isinstance(value, int):
                self.parser.add_argument(
                    arg_name,
                    type=int,
                    default=None,
                    help=f'Override for {key}'
                )
            elif isinstance(value, float):
                self.parser.add_argument(
                    arg_name,
                    type=float,
                    default=None,
                    help=f'Override for {key}'
                )
            else:
                self.parser.add_argument(
                    arg_name,
                    type=str,
                    default=None,
                    help=f'Override for {key}'
                )
                
    def parse_args(self, args=None):
        """Parse arguments and merge with config."""
        # First parse to get config file
        self.add_config_argument()
        known_args, remaining = self.parser.parse_known_args(args)
        
        # Load config if provided
        if known_args.config:
            config = load_yaml_config(known_args.config)
            config = expand_env_vars(config)
            
            # Add arguments based on config
            self.add_arguments_from_config(config)
            
            # Parse all arguments
            all_args = self.parser.parse_args(args)
            
            # Convert to dict and remove None values
            args_dict = {k: v for k, v in vars(all_args).items() if v is not None}
            
            # Remove the config argument itself
            args_dict.pop('config', None)
            
            # Merge with config (CLI overrides config)
            flat_config = flatten_config(config)
            
            # Update config with CLI overrides
            for key, value in args_dict.items():
                flat_key = key.replace('-', '_')
                if value is not None:
                    flat_config[flat_key] = value
                    
            # Return the flat config directly - don't unflatten
            # The trainer_config.py will handle the conversion
            final_config = flat_config
            
            return final_config
        else:
            # No config file, just parse regular arguments
            return vars(self.parser.parse_args(args))


def load_config(config_path: Optional[str] = None, cli_args: Optional[list] = None) -> Dict[str, Any]:
    """
    Main entry point for loading configuration.
    
    Args:
        config_path: Path to YAML config file
        cli_args: List of CLI arguments (for testing)
    
    Returns:
        Merged configuration dictionary
    """
    parser = ConfigArgumentParser()
    
    # If config_path is provided directly, inject it into args
    if config_path and cli_args is None:
        cli_args = ['--config', config_path]
    
    config = parser.parse_args(cli_args)
    
    # Handle special case for 'auto' values
    if config.get('data', {}).get('max_seq_length') == 'auto':
        config['data']['max_seq_length'] = None  # Will be detected from dataset
    
    return config


if __name__ == "__main__":
    # Example usage
    import json
    
    # Test loading a config
    config = load_config()
    print(json.dumps(config, indent=2))