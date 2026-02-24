"""
Logger for SA-MOO
Implements a logger that writes to both console and file
"""

import sys
from pathlib import Path
from typing import TextIO


class Logger:
    """
    Logger that redirects output to both console and file
    """
    
    def __init__(self, log_file_path: Path):
        """
        Initialize the logger
        
        Args:
            log_file_path: Path to the log file
        """
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', buffering=1)  # Line buffered
        
    def write(self, message: str):
        """
        Write message to both terminal and log file
        
        Args:
            message: Message to write
        """
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        """
        Flush both outputs
        """
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        """
        Close the log file
        """
        self.log_file.close()