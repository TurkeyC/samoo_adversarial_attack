"""
Attack Orchestrator for SA-MOO
High-level component that orchestrates the entire attack process
"""

from ..config.config_manager import ConfigManager
from ..algorithms.sa_moo_algorithm import SaMooAlgorithm


class AttackOrchestrator:
    """
    Attack Orchestrator for SA-MOO
    
    High-level component that orchestrates the entire attack process by coordinating
    all the different modules and components of the system.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the attack orchestrator
        
        Args:
            config_file: Optional path to a configuration file
        """
        self.config_manager = ConfigManager(config_file)
        self.algorithm = SaMooAlgorithm(self.config_manager.system_config)
        
    def run_attack(self):
        """
        Run the complete SA-MOO attack
        """
        # Print current configuration
        self.config_manager.print_config()
        
        # Execute the algorithm
        self.algorithm.run()