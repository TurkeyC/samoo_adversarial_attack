"""
Unit tests for the configuration system
"""

import unittest
from pathlib import Path
from ..config.config_manager import ConfigManager
from ..config.types import SystemConfig


class TestConfig(unittest.TestCase):
    """Test cases for the configuration system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
        
    def test_system_config_creation(self):
        """Test that system config is properly created"""
        config = self.config_manager.system_config
        self.assertIsInstance(config, SystemConfig)
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.sa_moo)
        self.assertIsNotNone(config.attack)
        
    def test_default_values(self):
        """Test that default values are properly set"""
        config = self.config_manager.system_config
        self.assertEqual(config.target_image_id, 7780)
        self.assertEqual(config.sa_moo.population_size, 2)
        self.assertEqual(config.sa_moo.num_generations, 1000)
        self.assertEqual(config.sa_moo.fixed_k, 24)
        self.assertFalse(config.attack.is_targeted_attack)
        self.assertEqual(config.attack.perturbation_mode, "rgb_sim")
        
    def test_path_resolution(self):
        """Test that paths are properly resolved"""
        config = self.config_manager.system_config
        self.assertIsInstance(config.model.weights_path, Path)
        self.assertIsInstance(config.model.data_root_dir, Path)
        self.assertIsInstance(config.model.output_dir, Path)
        
    def test_model_path_validation(self):
        """Test that model path validation works"""
        # This test expects that a model file exists at the default location
        # In a real scenario, we might mock this or create a dummy file
        config = self.config_manager.system_config
        # Skip this test if the model file doesn't exist (expected in a test environment)
        if not config.model.weights_path.exists():
            self.skipTest(f"Model weights file does not exist at {config.model.weights_path}")


if __name__ == '__main__':
    unittest.main()