"""
BicameralAGI Logger
==========================

Overview:
---------
This module's purpose is to assist with logging and debugging possible issues while developing

Author: Alan Hourmand
Date: 9/23/2024
"""

import logging
from logging.handlers import RotatingFileHandler
import os


class BicaLogging:
    def __init__(self, name, log_file='core.log', log_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Get the absolute path of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the project root (two levels up from utils)
        project_root = os.path.dirname(os.path.dirname(current_dir))
        # Set the log directory at the project root
        log_dir = os.path.join(project_root, 'logs')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        handler = RotatingFileHandler(os.path.join(log_dir, log_file), maxBytes=10 * 1024 * 1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, message, **kwargs):
        self.logger.info(f"{message} | {kwargs}" if kwargs else message)

    def warning(self, message, **kwargs):
        self.logger.warning(f"{message} | {kwargs}" if kwargs else message)

    def error(self, message, exc_info=False, **kwargs):
        self.logger.error(f"{message} | {kwargs}" if kwargs else message, exc_info=exc_info)


if __name__ == "__main__":
    logger = BicaLogging("BicaTest")
    logger.info("Test info message", extra="Some info")
    logger.warning("Test warning message", issue="Potential problem")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("Test error message", exc_info=True)
