import yaml
import logging
import os
import datetime
from . import logmanager

class PromptManager:
    """ 
    Manages system and user prompts for the AI assistant.
    """
    def __init__(self, system_prompt_path: str = "./configuration/system_prompt.yaml"):
        """
        Initialize the PromptManager with the system prompt.

        Args:
            system_prompt_path: Path to the YAML file containing the system prompt
        """
        self.log = logmanager.LogManager()
        self.log.add_logfile("promptmanager")
        self.log.write_log("promptmanager", logging.INFO, "PromptManager initialized")
        
        self.system_prompt = self.load_system_prompt(system_prompt_path)

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt.

        Returns:
            The system prompt as a string
        """
        return self.system_prompt
    
    def load_system_prompt(self, filepath: str) -> str:
        """
        Load the system prompt from a YAML configuration file.

        Args:
            filepath: Path to the YAML file
        Returns:
            The system prompt as a string
        """
        
        try:
            with open(filepath, "r") as file:
                config = yaml.safe_load(file)
                system_prompt = "\n".join(
                    [line for line in config.values() if isinstance(line, str)]
                )
                self.log.write_log("promptmanager", logging.INFO, f"Loaded system prompt from {filepath}")
                self.system_prompt = system_prompt
                return system_prompt
        except Exception as e:
            self.log.write_log("promptmanager", logging.ERROR,  f"Error loading system prompt: {str(e)}")
            return ""