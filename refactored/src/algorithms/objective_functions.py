"""
Objective Functions for SA-MOO
Implements the objective functions and dominance relations for multi-objective optimization
"""

import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import rgb2lab, deltaE_ciede2000
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from typing import Tuple, List, Optional, Dict, Any

from ..config.types import SystemConfig


class ObjectiveFunctions:
    """
    Objective Functions for SA-MOO
    
    Implements the objective functions needed for the multi-objective optimization:
    - Loss-based objectives (classification loss)
    - L2 norm (perturbation magnitude)
    - L0 norm (sparsity constraint)
    - Dominance relations for comparing solutions
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize objective functions with the given configuration
        
        Args:
            config: System configuration object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize LPIPS model
        self.loss_fn_vgg = lpips.LPIPS(net="vgg").to(self.device)
        
    def evaluate_objectives_batch(
        self,
        population: List[Tuple[np.ndarray, np.ndarray]],
        original_rgb: np.ndarray,
        original_v: np.ndarray,
        true_label: int,
        model: torch.nn.Module,
        is_targeted_attack: bool,
        target_class_id: Optional[int],
        perturbation_mode: str
    ) -> List[Tuple[bool, float, float, float]]:
        """
        Evaluate all objectives for a batch of individuals
        
        Args:
            population: List of (indices, perturbations) tuples
            original_rgb: Original RGB image
            original_v: Original V channel (for V-channel mode)
            true_label: True label of the original image
            model: Target model to attack
            is_targeted_attack: Whether this is a targeted attack
            target_class_id: Target class ID for targeted attacks
            perturbation_mode: How perturbations are applied
            
        Returns:
            List of objective tuples (is_adversarial, loss, l2_norm, l0_norm) for each individual
        """
        objective_values = []
        
        for individual in population:
            indices, perturbations = individual
            
            # Reconstruct perturbed image
            if perturbation_mode == "v_channel":
                noise_full = np.zeros_like(original_v, dtype=np.float32)
                noise_full.flat[indices] = perturbations
                original_hsv = rgb_to_hsv(original_rgb)
                perturbed_v = np.clip(original_hsv[:, :, 2] + noise_full, 0.0, 1.0)
                perturbed_hsv = np.stack([
                    original_hsv[:, :, 0],
                    original_hsv[:, :, 1],
                    perturbed_v
                ], axis=-1)
                perturbed_rgb = hsv_to_rgb(perturbed_hsv)
            else:
                noise_full = np.zeros_like(original_rgb, dtype=np.float32)
                noise_full.flat[indices] = perturbations
                perturbed_rgb = np.clip(original_rgb + noise_full, 0.0, 1.0)
                
            # Compute objectives
            is_adv, loss = self._evaluate_loss_objective(
                perturbed_rgb, true_label, model, is_targeted_attack, target_class_id
            )
            l2_norm = self._compute_l2_norm(perturbations)
            l0_norm = self._compute_l0_norm(perturbations)
            
            objective_values.append((is_adv, loss, l2_norm, l0_norm))
            
        return objective_values
        
    def _evaluate_loss_objective(
        self,
        perturbed_rgb: np.ndarray,
        true_label: int,
        model: torch.nn.Module,
        is_targeted_attack: bool,
        target_class_id: Optional[int]
    ) -> Tuple[bool, float]:
        """
        Evaluate the loss objective (classification loss)
        
        Args:
            perturbed_rgb: Perturbed RGB image
            true_label: True label of the original image
            model: Target model to attack
            is_targeted_attack: Whether this is a targeted attack
            target_class_id: Target class ID for targeted attacks
            
        Returns:
            Tuple of (is_adversarial, loss_value)
        """
        # Convert to tensor and move to device
        perturbed_tensor = torch.from_numpy(perturbed_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(perturbed_tensor)
            probs = torch.softmax(logits, dim=1)
            
            if is_targeted_attack:
                # For targeted attack, we want to maximize the probability of the target class
                target_prob = probs[0, target_class_id].item()
                # Loss is negative log probability of target class (we want to minimize this)
                loss = -np.log(max(target_prob, 1e-8))  # Add small epsilon to prevent log(0)
                
                # Is adversarial if predicted class is the target class
                predicted_class = torch.argmax(probs, dim=1).item()
                is_adversarial = (predicted_class == target_class_id)
            else:
                # For non-targeted attack, we want to minimize the probability of the true class
                true_prob = probs[0, true_label].item()
                # Loss is negative log probability of true class (we want to minimize this to make it smaller)
                loss = -np.log(max(1 - true_prob, 1e-8))  # We want to minimize prob of true class
                
                # Is adversarial if predicted class is different from true class
                predicted_class = torch.argmax(probs, dim=1).item()
                is_adversarial = (predicted_class != true_label)
                
        return is_adversarial, loss
        
    def _compute_l2_norm(self, perturbations: np.ndarray) -> float:
        """
        Compute the L2 norm of perturbations
        
        Args:
            perturbations: Array of perturbation values
            
        Returns:
            L2 norm value
        """
        return float(np.linalg.norm(perturbations))
        
    def _compute_l0_norm(self, perturbations: np.ndarray) -> int:
        """
        Compute the L0 norm of perturbations (count of non-zero elements)
        
        Args:
            perturbations: Array of perturbation values
            
        Returns:
            L0 norm value (int)
        """
        return int(np.count_nonzero(perturbations))
        
    def dominates(self, obj1: Tuple[bool, float, float, float], obj2: Tuple[bool, float, float, float]) -> bool:
        """
        Check if objective tuple obj1 dominates obj2 according to the configured dominance rules
        
        Args:
            obj1: First objective tuple (adversarial, loss, l2_norm, l0_norm)
            obj2: Second objective tuple (adversarial, loss, l2_norm, l0_norm)
            
        Returns:
            True if obj1 dominates obj2
        """
        # Check the dominance rules from the config
        # First check adversarial status
        if obj1[0] and not obj2[0]:
            return True
        elif not obj1[0] and obj2[0]:
            return False
        # If both are adversarial or both are not, compare other objectives
        elif obj1[0] and obj2[0]:
            # Both are adversarial, compare L2 and L0 norms (both should be minimized)
            # obj1 dominates if it has lower L2 and not higher L0, or lower L0 and not higher L2
            if obj1[2] < obj2[2] and obj1[3] <= obj2[3]:  # Lower L2, not higher L0
                return True
            elif obj1[3] < obj2[3] and obj1[2] <= obj2[2]:  # Lower L0, not higher L2
                return True
            elif obj1[2] < obj2[2] and obj1[3] < obj2[3]:  # Both lower
                return True
        else:
            # Both are not adversarial, compare loss (should be minimized)
            # The one with lower loss dominates
            if obj1[1] < obj2[1]:
                return True
            elif obj1[1] == obj2[1]:
                # If losses are equal, compare L2 norms
                if obj1[2] < obj2[2]:
                    return True
                    
        return False
        
    def non_dominated_sort(self, objective_values: List[Tuple[bool, float, float, float]]) -> List[List[int]]:
        """
        Perform non-dominated sorting on the objective values
        
        Args:
            objective_values: List of tuples representing (adversarial, loss, l2_norm, l0_norm)
            
        Returns:
            List of fronts, where each front is a list of indices
        """
        # Dominance comparison: individual i dominates j if:
        # 1. i is adversarial and j is not, OR
        # 2. both are adversarial or both non-adversarial, and i is better in at least one objective and not worse in others
        n = len(objective_values)
        S = [[] for _ in range(n)]  # Indices of solutions dominated by i
        n_p = [0 for _ in range(n)]  # Number of solutions that dominate i
        F = [[]]  # Fronts
        rank = [0 for _ in range(n)]  # Rank of each solution
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(objective_values[i], objective_values[j]):
                        S[i].append(j)
                    elif self.dominates(objective_values[j], objective_values[i]):
                        n_p[i] += 1
                        
            if n_p[i] == 0:
                rank[i] = 0
                F[0].append(i)
                
        # Find subsequent fronts
        front_idx = 0
        while F[front_idx]:
            Q = []
            for i in F[front_idx]:
                for j in S[i]:
                    n_p[j] -= 1
                    if n_p[j] == 0:
                        rank[j] = front_idx + 1
                        Q.append(j)
            front_idx += 1
            F.append(Q)
            
        # Remove empty last front
        if not F[-1]:
            F.pop()
            
        return F