import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'agent': {
        'name': 'default_agent',
        'description': 'Default agent configuration'
    },
    'environment': {
        'name': 'default_environment',
        'description': 'Default environment configuration'
    },
    'velocity_threshold': 0.5,
    'flow_theory_threshold': 0.7
}

# Define configuration classes
class ConfigType(Enum):
    AGENT = 'agent'
    ENVIRONMENT = 'environment'

class Config(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def validate(self):
        pass

class AgentConfig(Config):
    def __init__(self, name: str, description: str, velocity_threshold: float, flow_theory_threshold: float):
        super().__init__(name, description)
        self.velocity_threshold = velocity_threshold
        self.flow_theory_threshold = flow_theory_threshold

    def validate(self):
        if not isinstance(self.velocity_threshold, (int, float)) or self.velocity_threshold < 0 or self.velocity_threshold > 1:
            raise ValueError('Velocity threshold must be a float between 0 and 1')
        if not isinstance(self.flow_theory_threshold, (int, float)) or self.flow_theory_threshold < 0 or self.flow_theory_threshold > 1:
            raise ValueError('Flow theory threshold must be a float between 0 and 1')

class EnvironmentConfig(Config):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)

    def validate(self):
        pass

# Define configuration manager
class ConfigurationManager:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict[ConfigType, Config]:
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                if config_data is None:
                    config_data = DEFAULT_CONFIG
                return self._parse_config(config_data)
        except FileNotFoundError:
            logger.warning(f'Config file not found: {self.config_file}')
            return DEFAULT_CONFIG

    def _parse_config(self, config_data: Dict) -> Dict[ConfigType, Config]:
        agent_config = AgentConfig(
            name=config_data['agent']['name'],
            description=config_data['agent']['description'],
            velocity_threshold=config_data['agent']['velocity_threshold'],
            flow_theory_threshold=config_data['agent']['flow_theory_threshold']
        )
        environment_config = EnvironmentConfig(
            name=config_data['environment']['name'],
            description=config_data['environment']['description']
        )
        return {
            ConfigType.AGENT: agent_config,
            ConfigType.ENVIRONMENT: environment_config
        }

    def save_config(self, config: Dict[ConfigType, Config]):
        with open(self.config_file, 'w') as f:
            yaml.dump(self._serialize_config(config), f, default_flow_style=False)

    def _serialize_config(self, config: Dict[ConfigType, Config]) -> Dict:
        return {
            'agent': {
                'name': config[ConfigType.AGENT].name,
                'description': config[ConfigType.AGENT].description,
                'velocity_threshold': config[ConfigType.AGENT].velocity_threshold,
                'flow_theory_threshold': config[ConfigType.AGENT].flow_theory_threshold
            },
            'environment': {
                'name': config[ConfigType.ENVIRONMENT].name,
                'description': config[ConfigType.ENVIRONMENT].description
            }
        }

# Define configuration context manager
@contextmanager
def config_context(config_manager: ConfigurationManager):
    try:
        yield config_manager.config
    except Exception as e:
        logger.error(f'Error occurred while accessing configuration: {e}')
        raise

# Define configuration loader
def load_config() -> Dict[ConfigType, Config]:
    config_manager = ConfigurationManager()
    with config_context(config_manager) as config:
        return config

# Define configuration saver
def save_config(config: Dict[ConfigType, Config]):
    config_manager = ConfigurationManager()
    config_manager.save_config(config)

# Example usage
if __name__ == '__main__':
    config = load_config()
    logger.info(f'Loaded configuration: {config}')
    # Modify configuration
    config[ConfigType.AGENT].velocity_threshold = 0.6
    save_config(config)
    logger.info(f'Saved configuration: {config}')