"""
Unit tests for the data module
"""

import unittest
import numpy as np
from ..config.config_manager import ConfigManager
from ..data.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for the data loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        config_manager = ConfigManager()
        # Override the model weights path to a dummy value to avoid errors
        config_manager.system_config.model.weights_path = "/tmp/dummy.pth"
        self.data_loader = DataLoader(config_manager.system_config)
        
    def test_class_names(self):
        """Test that CIFAR-10 class names are correctly loaded"""
        expected_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.assertEqual(self.data_loader.class_names, expected_classes)
        
    def test_load_model_failure_handling(self):
        """Test that model loading handles missing files gracefully"""
        # This test verifies that the model loading method can handle missing weight files
        model = self.data_loader._load_model()
        # Should return a model even if weights couldn't be loaded
        self.assertIsNotNone(model)
        # Model should be a ResNet-18 variant
        self.assertTrue(hasattr(model, 'fc'))  # Check for classifier layer


if __name__ == '__main__':
    unittest.main()