"""
Main SA-MOO Algorithm Implementation
This module implements the core SA-MOO (Sparse Adversarial Multi-Objective Optimization) algorithm
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from ..config.types import SystemConfig
from ..utils.logger import Logger
from ..utils.edge_guidance import EdgeGuidanceProcessor
from ..data.data_loader import DataLoader
from ..algorithms.objective_functions import ObjectiveFunctions
from ..algorithms.evolutionary_operators import EvolutionaryOperators
from ..visualization.visualizer import Visualizer


class SaMooAlgorithm:
    """
    SA-MOO (Sparse Adversarial Multi-Objective Optimization) Algorithm
    
    Implements the complete adversarial attack pipeline using multi-objective optimization
    to generate sparse adversarial examples that minimize both perturbation magnitude 
    and pixel count while maximizing misclassification success.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the SA-MOO algorithm with the given configuration
        
        Args:
            config: System configuration object containing all parameters
        """
        self.config = config
        self.logger = None
        self.visualizer = None
        self.edge_guidance_processor = None
        
        # Initialize components
        self.objective_functions = ObjectiveFunctions(config)
        self.evolutionary_operators = EvolutionaryOperators(config)
        self.data_loader = DataLoader(config)
        
        # Runtime attributes
        self.original_rgb = None
        self.original_v = None
        self.true_label = None
        self.model = None
        self.class_names = None
        self.run_dir = None
        
    def setup_run_environment(self):
        """Set up the run environment including logging and output directories"""
        # Create run directory
        time_str = datetime.now().strftime("%m%d%H%M")
        attack_type = "targeted" if self.config.attack.is_targeted_attack else "non_targeted"
        run_name = f"img{self.config.target_image_id}_{attack_type}_k{self.config.sa_moo.fixed_k}_{self.config.attack.perturbation_mode}_gen{self.config.sa_moo.num_generations}_{time_str}"
        self.run_dir = self.config.model.output_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = Logger(self.run_dir / "report.txt")
        import sys
        sys.stdout = self.logger  # Redirect output to log file
        
        # Filter warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
        warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision.models._utils")
        
    def setup_components(self):
        """Set up all algorithm components"""
        # Load data and model
        self.original_rgb, self.original_v, self.true_label, self.model, self.class_names = \
            self.data_loader.load_target_image_and_model(self.config.target_image_id)
            
        # Initialize edge guidance processor if enabled
        if self.config.edge_guidance.enabled:
            self.edge_guidance_processor = EdgeGuidanceProcessor(self.config)
            
        # Initialize visualizer if enabled
        if self.config.visualization.enable:
            self.visualizer = Visualizer(self.original_rgb)
            
    def run(self):
        """Execute the complete SA-MOO algorithm"""
        try:
            # Setup environment
            self.setup_run_environment()
            self.setup_components()
            
            # Print configuration
            print("--- Starting Unified SA-MOO Attack ---")
            self.print_configuration()
            
            # Initialize population
            population = self.evolutionary_operators.initialize_population(
                pop_size=self.config.sa_moo.population_size,
                fixed_k=self.config.sa_moo.fixed_k,
                perturbation_mode=self.config.attack.perturbation_mode,
                zero_sample_prob=self.config.sa_moo.zero_sample_prob,
                edge_guidance=self.edge_guidance_processor.get_weights(self.original_rgb) if self.edge_guidance_processor else None
            )
            
            # Initialize tracking variables
            best_solution = None
            best_objectives = (False, float("inf"), float("inf"), float("inf"))
            
            # History for convergence plot
            history = {"l2": [], "l0": []}
            
            # Main evolutionary loop
            for generation in range(self.config.sa_moo.num_generations):
                # Generate offspring through crossover
                offspring = self._generate_offspring(population, generation)
                
                # Apply mutation
                mutated_offspring = self._apply_mutation(offspring, generation)
                
                # Combine parents and offspring
                combined = population + mutated_offspring
                
                # Evaluate objectives
                objective_values = self.objective_functions.evaluate_objectives_batch(
                    combined,
                    self.original_rgb,
                    self.original_v,
                    self.true_label,
                    self.model,
                    self.config.attack.is_targeted_attack,
                    self.config.attack.target_class_id if self.config.attack.is_targeted_attack else None,
                    self.config.attack.perturbation_mode
                )
                
                # Select next generation
                population = self.evolutionary_operators.selection(
                    combined,
                    objective_values,
                    self.config.sa_moo.population_size
                )
                
                # Update best solution
                current_objectives = self.objective_functions.evaluate_objectives_batch(
                    population,
                    self.original_rgb,
                    self.original_v,
                    self.true_label,
                    self.model,
                    self.config.attack.is_targeted_attack,
                    self.config.attack.target_class_id if self.config.attack.is_targeted_attack else None,
                    self.config.attack.perturbation_mode
                )
                
                # Get best individual from first front
                fronts = self.objective_functions.non_dominated_sort(current_objectives)
                best_idx = fronts[0][0] if fronts[0] else 0
                
                if self.objective_functions.dominates(current_objectives[best_idx], best_objectives):
                    best_solution = population[best_idx]
                    best_objectives = current_objectives[best_idx]
                    
                # Record history
                if best_objectives[0]:  # If successful attack
                    history["l2"].append(best_objectives[2])
                    history["l0"].append(best_objectives[3])
                else:
                    history["l2"].append(None)
                    history["l0"].append(None)
                    
                # Print generation info
                self._print_generation_info(generation, best_objectives)
                
                # Update visualization
                if self.config.visualization.enable and (generation + 1) % self.config.visualization.interval == 0:
                    self.visualizer.update(
                        self.original_rgb,
                        best_solution,
                        best_objectives,
                        generation + 1,
                        self.config.attack.perturbation_mode,
                        self.original_v
                    )
                    
            # Finalize results
            self._finalize_results(best_solution, best_objectives, history)
            
        finally:
            # Clean up
            self._cleanup()
    
    def _generate_offspring(self, population: List[Tuple[np.ndarray, np.ndarray]], generation: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate offspring through crossover operation"""
        offspring = []
        while len(offspring) < self.config.sa_moo.population_size:
            # Randomly select two parents
            i1, i2 = np.random.choice(len(population), 2, replace=False)
            p1, p2 = population[i1], population[i2]
            
            if np.random.rand() < self.config.sa_moo.crossover_prob:
                # Perform crossover
                c1, c2 = self.evolutionary_operators.crossover(
                    p1, p2, 
                    self.config.sa_moo.fixed_k, 
                    self.config.attack.perturbation_mode,
                    self.config.sa_moo.crossover_prob
                )
                offspring.extend([c1, c2])
            else:
                # No crossover, just copy parents
                offspring.extend([p1, p2])
                
        return offspring[:self.config.sa_moo.population_size]  # Ensure correct size
    
    def _apply_mutation(self, offspring: List[Tuple[np.ndarray, np.ndarray]], generation: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Apply mutation to offspring"""
        # Calculate dynamic mutation probability based on generation
        progress = generation / max(self.config.sa_moo.num_generations - 1, 1)
        fallback_start_pm = 0.4 if not self.config.attack.is_targeted_attack else 0.2
        current_pm = max(0.001, fallback_start_pm * (1 - progress))
        
        # Adjust based on dynamic parameter schedule if configured
        if "mutation_probability" in self.config.dynamic_parameters.mutation_probability:
            # Apply dynamic scheduling logic here
            pass
        
        return [
            self.evolutionary_operators.mutate(
                individual,
                self.config.sa_moo.fixed_k,
                current_pm,
                self.config.attack.perturbation_mode,
                self.config.sa_moo.zero_sample_prob,
                edge_guidance=self.edge_guidance_processor.get_weights(self.original_rgb) if self.edge_guidance_processor else None
            )
            for individual in offspring
        ]
    
    def _print_generation_info(self, generation: int, best_objectives: Tuple[bool, float, float, float]):
        """Print information about the current generation"""
        print(
            f"Gen {generation+1}/{self.config.sa_moo.num_generations} | "
            f"Best L2: {best_objectives[2]:.2f}, "
            f"L0: {best_objectives[3]}, Success: {best_objectives[0]}"
        )
    
    def print_configuration(self):
        """Print the current configuration"""
        print(f"Perturbation Mode: {self.config.attack.perturbation_mode}")
        print(f"Attack Type: {'Targeted' if self.config.attack.is_targeted_attack else 'Non-Targeted'}")
        if self.config.attack.is_targeted_attack:
            print(f"Target Class: {self.class_names[self.config.attack.target_class_id]} ({self.config.attack.target_class_id})")
        print(f"Fixed K (L0): {self.config.sa_moo.fixed_k}")
        print(f"Generations: {self.config.sa_moo.num_generations}, Population Size: {self.config.sa_moo.population_size}")
        print(f"Crossover Prob: {self.config.sa_moo.crossover_prob}")
        print(f"Zero-sample Prob: {self.config.sa_moo.zero_sample_prob}")
        print(f"Visualization: {'ENABLED' if self.config.visualization.enable else 'DISABLED'}")
        print(f"Real World Robustness: {'ENABLED' if self.config.real_world_robustness.enable else 'DISABLED'}")
        if self.config.real_world_robustness.enable:
            print(f"  JPEG Quality: {self.config.real_world_robustness.jpeg_quality}, "
                  f"Resize: {'ENABLED' if self.config.real_world_robustness.enable_resize_preprocessing else 'DISABLED'}")
            if self.config.real_world_robustness.enable_resize_preprocessing:
                print(f"  Resize Scale: {self.config.real_world_robustness.resize_scale}")
        print(f"Edge Guidance: {'ENABLED' if self.config.edge_guidance.enabled else 'DISABLED'}")
        if self.config.edge_guidance.enabled:
            print(f"  Method: {self.config.edge_guidance.method}")
    
    def _finalize_results(self, best_solution: Tuple[np.ndarray, np.ndarray], best_objectives: Tuple[bool, float, float, float], history: Dict[str, List]):
        """Process and save final results"""
        print("\n--- Attack Completed ---")
        
        if best_solution is None:
            print("Failed to find an adversarial example.")
            return
            
        # Reconstruct final perturbed image
        indices, perturbations = best_solution
        if self.config.attack.perturbation_mode == "v_channel":
            noise_full = np.zeros_like(self.original_v, dtype=np.float32)
            noise_full.flat[indices] = perturbations
            original_hsv = rgb_to_hsv(self.original_rgb)
            final_v = np.clip(original_hsv[:, :, 2] + noise_full, 0.0, 1.0)
            final_hsv = np.stack([
                original_hsv[:, :, 0],
                original_hsv[:, :, 1],
                final_v
            ], axis=-1)
            final_rgb = hsv_to_rgb(final_hsv)
        else:
            noise_full = np.zeros_like(self.original_rgb, dtype=np.float32)
            noise_full.flat[indices] = perturbations
            final_rgb = np.clip(self.original_rgb + noise_full, 0.0, 1.0)
            
        # Save results using visualizer
        if self.visualizer:
            self.visualizer.save_results(
                self.original_rgb,
                noise_full,
                final_rgb,
                self.true_label,
                self.model,
                self.class_names,
                best_objectives,
                self.config.attack.perturbation_mode,
                self.run_dir
            )
        
        # Save convergence plot
        if self.visualizer:
            self.visualizer.save_convergence_plot(
                history["l2"], 
                history["l0"], 
                self.run_dir / "convergence.png"
            )
        
        # Print final statistics
        print("Final Best Solution Stats:")
        print(f"  Adversarial: {best_objectives[0]}")
        print(f"  L2 Norm: {best_objectives[2]:.4f}")
        print(f"  L0 Norm: {best_objectives[3]}")
        
        print(f"\nResults saved in: {self.run_dir.absolute()}")
        
    def _cleanup(self):
        """Clean up resources"""
        import sys
        if self.logger:
            # Restore original stdout
            sys.stdout = self.logger.terminal
            self.logger.close()
            
        if self.visualizer:
            self.visualizer.cleanup()