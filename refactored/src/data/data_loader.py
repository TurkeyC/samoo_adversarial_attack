"""
Data Loader for SA-MOO
Handles loading of images, models, and datasets for the adversarial attack
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision.datasets import CIFAR10
import numpy as np
from PIL import Image
from typing import Tuple, Optional

from ..config.types import SystemConfig


class DataLoader:
    """
    Data Loader for SA-MOO
    
    Handles loading of target images, models, and datasets needed for the adversarial attack.
    Provides utilities for converting between different color spaces and preparing data
    for the neural network.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the data loader with the given configuration
        
        Args:
            config: System configuration object
        """
        self.config = config
        
        # Define transformations for CIFAR-10
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # CIFAR-10 class names
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
    def load_target_image_and_model(
        self, 
        target_image_id: int
    ) -> Tuple[np.ndarray, np.ndarray, int, torch.nn.Module, list]:
        """
        Load the target image and model for the attack
        
        Args:
            target_image_id: Index of the target image in the CIFAR-10 test set
            
        Returns:
            Tuple containing:
            - original_rgb: Original RGB image (normalized to [0, 1])
            - original_v: Original V channel (for V-channel mode)
            - true_label: True label of the image
            - model: Loaded model ready for inference
            - class_names: List of class names
        """
        # Load CIFAR-10 test dataset
        test_dataset = CIFAR10(root=str(self.config.model.data_root_dir), train=False, download=True, transform=self.transform)
        
        # Get the target image
        if target_image_id >= len(test_dataset):
            raise ValueError(f"Target image ID {target_image_id} is out of range for CIFAR-10 test set (size: {len(test_dataset)})")
            
        image_tensor, true_label = test_dataset[target_image_id]
        original_pil = test_dataset.data[target_image_id]  # Original image as numpy array
        
        # Convert to RGB normalized to [0, 1]
        original_rgb = original_pil.astype(np.float32) / 255.0
        
        # Extract V channel if needed
        from matplotlib.colors import rgb_to_hsv
        original_hsv = rgb_to_hsv(original_rgb)
        original_v = original_hsv[:, :, 2]  # V channel
        
        # Load the model
        model = self._load_model()
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        return original_rgb, original_v, true_label, model, self.class_names
        
    def _load_model(self) -> torch.nn.Module:
        """
        Load the target model for the attack
        
        Returns:
            Loaded model ready for inference
        """
        # Load a pretrained ResNet-18 model (or load from custom weights if provided)
        try:
            # First try to load from custom weights if provided
            if self.config.model.weights_path.exists():
                model = resnet18(num_classes=10)  # CIFAR-10 has 10 classes
                checkpoint = torch.load(str(self.config.model.weights_path), map_location='cpu')
                
                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Use torchvision's pretrained model if custom weights don't exist
                print(f"Warning: Custom weights not found at {self.config.model.weights_path}, using standard ResNet-18")
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                # Modify the final layer to match CIFAR-10's 10 classes
                model.fc = torch.nn.Linear(model.fc.in_features, 10)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new ResNet-18 model with random weights...")
            model = resnet18(num_classes=10)  # CIFAR-10 has 10 classes
            
        return model