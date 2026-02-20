import logging
from operator import ne
import os
import wandb
from typing import Dict

# Ensure Logs folder exists
os.makedirs('Logs', exist_ok=True)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Important: capture all levels
logger.propagate = False  # Prevent log messages from being propagated to the root logger

# Console handler (optional)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler('Logs/debug.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

class Logger:
    def __init__(self,run:wandb.Run=None):
        self.run = run
        self.logger = logger

    def log(self, data:Dict, step=None):
        if self.run is not None:
            self.run.log(data, step=step)
        self.logger.log(logging.INFO,data.__str__())

    def info(self, message):
        self.logger.info(message)

logger_ = Logger()

def get_logger(run:wandb.Run=None):
    return Logger(run)