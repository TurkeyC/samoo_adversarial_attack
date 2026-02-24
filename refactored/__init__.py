"""
Refactored SA-MOO Adversarial Attack Package
A clean, modular implementation of the SA-MOO algorithm for generating sparse adversarial examples.
"""

__version__ = "1.0.0"
__author__ = "SA-MOO Implementation Team"

from .src.components.attack_orchestrator import AttackOrchestrator
from .src.config.config_manager import ConfigManager

__all__ = ['AttackOrchestrator', 'ConfigManager']