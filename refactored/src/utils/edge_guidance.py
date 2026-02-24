"""
Edge Guidance Processor for SA-MOO
Implements edge-based guidance for initializing and mutating individuals
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation
from skimage.filters import sobel, scharr, prewitt, canny
from skimage.feature import canny as canny_skimage
from skimage.morphology import dilation, erosion
from skimage.transform import resize
from typing import Optional, Dict, Any

from ..config.types import SystemConfig


class EdgeGuidanceProcessor:
    """
    Edge Guidance Processor for SA-MOO
    
    Implements edge-based guidance for initializing and mutating individuals.
    Uses various edge detection methods and allows for multi-scale analysis
    and semantic guidance fusion.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the edge guidance processor with the given configuration
        
        Args:
            config: System configuration object
        """
        self.config = config
        
    def get_weights(self, image: np.ndarray) -> np.ndarray:
        """
        Compute edge guidance weights for the given image
        
        Args:
            image: Input image (H, W, C) in range [0, 1]
            
        Returns:
            Edge guidance weights as a 2D array of the same spatial dimensions as input
        """
        if not self.config.edge_guidance.enabled:
            # Return uniform weights if edge guidance is disabled
            return np.ones(image.shape[:2]) / (image.shape[0] * image.shape[1])
            
        # Convert to grayscale for edge detection
        if image.ndim == 3 and image.shape[2] == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image.squeeze()  # Assume it's already grayscale
            
        # Apply Gaussian smoothing if specified
        if self.config.edge_guidance.gaussian_sigma > 0:
            gray = ndimage.gaussian_filter(gray, sigma=self.config.edge_guidance.gaussian_sigma)
            
        # Compute edge map based on the specified method
        edge_map = self._compute_edge_map(gray)
        
        # Apply multi-scale processing if enabled
        if self.config.edge_guidance.multi_scale["enabled"]:
            edge_map = self._apply_multi_scale_processing(edge_map)
            
        # Apply semantic guidance if enabled
        if self.config.edge_guidance.semantic["enabled"]:
            edge_map = self._fuse_with_semantic_guidance(edge_map, image)
            
        # Apply exponentiation to increase contrast
        edge_map = np.power(edge_map + self.config.edge_guidance.min_value, self.config.edge_guidance.exponent)
        
        # Mix with uniform distribution
        uniform_weight = self.config.edge_guidance.uniform_mix
        edge_map = (1 - uniform_weight) * edge_map + uniform_weight * np.mean(edge_map)
        
        # Normalize so that probabilities sum to 1
        edge_map = edge_map / edge_map.sum()
        
        return edge_map
        
    def _compute_edge_map(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Compute edge map using the specified method
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Edge map
        """
        method = self.config.edge_guidance.method
        edge_map = None
        
        if method == "sobel":
            edge_map = sobel(gray_image)
        elif method == "scharr":
            edge_map = scharr(gray_image)
        elif method == "prewitt":
            edge_map = prewitt(gray_image)
        elif method == "canny":
            sigma = self.config.edge_guidance.canny_sigma
            edge_map = canny_skimage(
                gray_image, 
                sigma=sigma,
                low_threshold=0.1,
                high_threshold=0.2
            ).astype(float)
        else:
            # Default to sobel if unknown method
            edge_map = sobel(gray_image)
            
        # Ensure non-negative values
        edge_map = np.abs(edge_map)
        
        return edge_map
        
    def _apply_multi_scale_processing(self, edge_map: np.ndarray) -> np.ndarray:
        """
        Apply multi-scale edge processing
        
        Args:
            edge_map: Base edge map
            
        Returns:
            Multi-scale processed edge map
        """
        scales = self.config.edge_guidance.multi_scale["scales"]
        aggregation_method = self.config.edge_guidance.multi_scale["combine"]
        
        aggregated_map = np.zeros_like(edge_map)
        total_weight = 0
        
        for scale_factor in scales:
            if scale_factor == 1.0:
                scaled_map = edge_map
            else:
                # Resize the edge map
                new_shape = (int(edge_map.shape[0] * scale_factor), int(edge_map.shape[1] * scale_factor))
                scaled_map = resize(edge_map, new_shape, anti_aliasing=True, preserve_range=True)
                
                # Resize back to original size for aggregation
                scaled_map = resize(scaled_map, edge_map.shape, anti_aliasing=True, preserve_range=True)
                
            # Apply Gaussian smoothing if specified
            sigma = self.config.edge_guidance.multi_scale["gaussian_sigmas"]
            if sigma is not None:
                if isinstance(sigma, (list, tuple)):
                    # Use per-scale sigma values
                    idx = scales.index(scale_factor)
                    if idx < len(sigma):
                        scaled_map = ndimage.gaussian_filter(scaled_map, sigma=sigma[idx])
                else:
                    scaled_map = ndimage.gaussian_filter(scaled_map, sigma=sigma)
                    
            # Aggregate using the specified method
            if aggregation_method == "max":
                aggregated_map = np.maximum(aggregated_map, scaled_map)
            elif aggregation_method == "mean":
                aggregated_map = aggregated_map + scaled_map
                total_weight += 1
            elif aggregation_method == "sum":
                aggregated_map = aggregated_map + scaled_map
            else:
                # Default to mean
                aggregated_map = aggregated_map + scaled_map
                total_weight += 1
                
        if aggregation_method == "mean":
            aggregated_map = aggregated_map / total_weight if total_weight > 0 else aggregated_map
            
        return aggregated_map
        
    def _fuse_with_semantic_guidance(self, edge_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Fuse edge map with semantic guidance
        
        Args:
            edge_map: Base edge map
            image: Original image for reference
            
        Returns:
            Fused edge-semantic guidance map
        """
        # For now, we'll just return the edge map
        # In a full implementation, we would load and fuse with semantic maps
        return edge_map