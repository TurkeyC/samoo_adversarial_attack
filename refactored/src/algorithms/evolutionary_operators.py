"""
Evolutionary Operators for SA-MOO
Implements the core evolutionary operators: initialization, crossover, mutation, and selection
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from scipy.ndimage import binary_dilation

from ..config.types import SystemConfig


class EvolutionaryOperators:
    """
    Evolutionary Operators for SA-MOO
    
    Implements the core evolutionary operators needed for the multi-objective optimization:
    - Population initialization
    - Crossover operations
    - Mutation operations
    - Selection mechanisms
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize evolutionary operators with the given configuration
        
        Args:
            config: System configuration object
        """
        self.config = config
        
    def initialize_population(
        self, 
        pop_size: int, 
        fixed_k: int, 
        perturbation_mode: str, 
        zero_sample_prob: float = 0.3,
        edge_guidance: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Initialize a population of individuals for the evolutionary algorithm
        
        Args:
            pop_size: Size of the population
            fixed_k: Number of perturbations allowed (L0 constraint)
            perturbation_mode: How perturbations are applied ('rgb_sim', 'channel', 'v_channel')
            zero_sample_prob: Probability of sampling zero for perturbation
            edge_guidance: Optional edge guidance weights for initialization
            
        Returns:
            Initial population as a list of (indices, perturbations) tuples
        """
        population = []
        
        for _ in range(pop_size):
            # Determine the shape based on perturbation mode
            if perturbation_mode == "v_channel":
                # For V-channel, we work with flattened 32*32 array
                flat_size = 32 * 32
            elif perturbation_mode == "channel":
                # For channel mode, we work with flattened 32*32*3 array
                flat_size = 32 * 32 * 3
            elif perturbation_mode == "rgb_sim":
                # For RGB simultaneous, we work with flattened 32*32 array
                flat_size = 32 * 32
            else:
                raise ValueError(f"Unknown perturbation mode: {perturbation_mode}")
                
            # Sample k random indices
            if edge_guidance is not None:
                # Use edge guidance for weighted sampling
                flat_weights = edge_guidance.flatten()
                # Normalize weights
                flat_weights = flat_weights / flat_weights.sum()
                indices = np.random.choice(flat_size, size=fixed_k, replace=False, p=flat_weights)
            else:
                # Uniform random sampling
                indices = np.random.choice(flat_size, size=fixed_k, replace=False)
                
            # Generate perturbations based on mode
            if self.config.continuous_perturbation.enable:
                # Continuous perturbations
                perturbations = np.random.uniform(
                    low=self.config.continuous_perturbation.lower_bound,
                    high=self.config.continuous_perturbation.upper_bound,
                    size=fixed_k
                )
                # Round to specified decimal places
                perturbations = np.round(perturbations, self.config.continuous_perturbation.decimal_places)
            else:
                # Discrete perturbations {-1, 0, 1}
                perturbations = np.random.choice([-1, 0, 1], size=fixed_k, p=[0.35, 0.3, 0.35])
                
                # Apply zero sampling probability
                zero_mask = np.random.random(size=fixed_k) < zero_sample_prob
                perturbations[zero_mask] = 0
                
                # Ensure at least some non-zero values to avoid trivial solutions
                if np.all(perturbations == 0):
                    # Randomly select one position to be non-zero
                    idx = np.random.randint(0, len(perturbations))
                    perturbations[idx] = np.random.choice([-1, 1])
                    
            # Sort indices to maintain consistency
            sorted_idx = np.argsort(indices)
            indices = indices[sorted_idx]
            perturbations = perturbations[sorted_idx]
            
            population.append((indices.copy(), perturbations.copy()))
            
        return population
        
    def crossover(
        self, 
        parent1: Tuple[np.ndarray, np.ndarray], 
        parent2: Tuple[np.ndarray, np.ndarray], 
        fixed_k: int, 
        perturbation_mode: str,
        crossover_prob: float
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Perform crossover between two parent individuals
        
        Args:
            parent1: First parent (indices, perturbations)
            parent2: Second parent (indices, perturbations)
            fixed_k: Number of perturbations allowed
            perturbation_mode: How perturbations are applied
            crossover_prob: Crossover probability
            
        Returns:
            Two offspring individuals
        """
        indices1, perturbations1 = parent1
        indices2, perturbations2 = parent2
        
        # Simple crossover: combine indices and perturbations from both parents
        # Get union of indices from both parents
        all_indices = np.union1d(indices1, indices2)
        
        # Randomly assign perturbations from either parent for each index
        child1_indices = []
        child1_perturbations = []
        child2_indices = []
        child2_perturbations = []
        
        for idx in all_indices:
            # Find positions in each parent
            pos1 = np.where(indices1 == idx)[0]
            pos2 = np.where(indices2 == idx)[0]
            
            if len(pos1) > 0 and len(pos2) > 0:
                # Index present in both parents
                if np.random.rand() < 0.5:
                    child1_indices.append(idx)
                    child1_perturbations.append(perturbations1[pos1[0]])
                    child2_indices.append(idx)
                    child2_perturbations.append(perturbations2[pos2[0]])
                else:
                    child1_indices.append(idx)
                    child1_perturbations.append(perturbations2[pos2[0]])
                    child2_indices.append(idx)
                    child2_perturbations.append(perturbations1[pos1[0]])
            elif len(pos1) > 0:
                # Only in parent1
                child1_indices.append(idx)
                child1_perturbations.append(perturbations1[pos1[0]])
                if np.random.rand() < crossover_prob:
                    child2_indices.append(idx)
                    child2_perturbations.append(perturbations1[pos1[0]])
            elif len(pos2) > 0:
                # Only in parent2
                if np.random.rand() < crossover_prob:
                    child1_indices.append(idx)
                    child1_perturbations.append(perturbations2[pos2[0]])
                child2_indices.append(idx)
                child2_perturbations.append(perturbations2[pos2[0]])
                
        # Ensure each child has exactly fixed_k elements
        child1_indices = np.array(child1_indices)
        child1_perturbations = np.array(child1_perturbations)
        child2_indices = np.array(child2_indices)
        child2_perturbations = np.array(child2_perturbations)
        
        # If too many elements, randomly select fixed_k
        if len(child1_indices) > fixed_k:
            selected = np.random.choice(len(child1_indices), size=fixed_k, replace=False)
            child1_indices = child1_indices[selected]
            child1_perturbations = child1_perturbations[selected]
        elif len(child1_indices) < fixed_k:
            # Add random indices to reach fixed_k
            remaining_k = fixed_k - len(child1_indices)
            if perturbation_mode == "v_channel":
                flat_size = 32 * 32
            elif perturbation_mode == "channel":
                flat_size = 32 * 32 * 3
            elif perturbation_mode == "rgb_sim":
                flat_size = 32 * 32
            else:
                raise ValueError(f"Unknown perturbation mode: {perturbation_mode}")
                
            available_indices = np.setdiff1d(np.arange(flat_size), child1_indices)
            if len(available_indices) >= remaining_k:
                new_indices = np.random.choice(available_indices, size=remaining_k, replace=False)
            else:
                new_indices = available_indices  # Use all available if not enough
                
            new_perturbations = np.zeros(len(new_indices))
            child1_indices = np.concatenate([child1_indices, new_indices])
            child1_perturbations = np.concatenate([child1_perturbations, new_perturbations])
            
        if len(child2_indices) > fixed_k:
            selected = np.random.choice(len(child2_indices), size=fixed_k, replace=False)
            child2_indices = child2_indices[selected]
            child2_perturbations = child2_perturbations[selected]
        elif len(child2_indices) < fixed_k:
            # Add random indices to reach fixed_k
            remaining_k = fixed_k - len(child2_indices)
            if perturbation_mode == "v_channel":
                flat_size = 32 * 32
            elif perturbation_mode == "channel":
                flat_size = 32 * 32 * 3
            elif perturbation_mode == "rgb_sim":
                flat_size = 32 * 32
            else:
                raise ValueError(f"Unknown perturbation mode: {perturbation_mode}")
                
            available_indices = np.setdiff1d(np.arange(flat_size), child2_indices)
            if len(available_indices) >= remaining_k:
                new_indices = np.random.choice(available_indices, size=remaining_k, replace=False)
            else:
                new_indices = available_indices  # Use all available if not enough
                
            new_perturbations = np.zeros(len(new_indices))
            child2_indices = np.concatenate([child2_indices, new_indices])
            child2_perturbations = np.concatenate([child2_perturbations, new_perturbations])
            
        # Sort indices to maintain consistency
        sorted_idx1 = np.argsort(child1_indices)
        child1_indices = child1_indices[sorted_idx1]
        child1_perturbations = child1_perturbations[sorted_idx1]
        
        sorted_idx2 = np.argsort(child2_indices)
        child2_indices = child2_indices[sorted_idx2]
        child2_perturbations = child2_perturbations[sorted_idx2]
        
        return (
            (child1_indices.copy(), child1_perturbations.copy()),
            (child2_indices.copy(), child2_perturbations.copy())
        )
        
    def mutate(
        self, 
        individual: Tuple[np.ndarray, np.ndarray], 
        fixed_k: int, 
        mutation_prob: float, 
        perturbation_mode: str,
        zero_sample_prob: float = 0.3,
        edge_guidance: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mutation to an individual
        
        Args:
            individual: Individual to mutate (indices, perturbations)
            fixed_k: Number of perturbations allowed
            mutation_prob: Probability of mutation per element
            perturbation_mode: How perturbations are applied
            zero_sample_prob: Probability of sampling zero for perturbation
            edge_guidance: Optional edge guidance weights for mutation
            
        Returns:
            Mutated individual
        """
        indices, perturbations = individual
        
        # Copy to avoid modifying original
        new_indices = indices.copy()
        new_perturbations = perturbations.copy()
        
        # Determine if we should change indices
        if np.random.rand() < mutation_prob:
            # Change some indices based on mutation rate
            num_to_change = max(1, int(len(indices) * 0.1))  # Change 10% of indices
            
            # Randomly select positions to change
            positions_to_change = np.random.choice(len(indices), size=num_to_change, replace=False)
            
            if perturbation_mode == "v_channel":
                flat_size = 32 * 32
            elif perturbation_mode == "channel":
                flat_size = 32 * 32 * 3
            elif perturbation_mode == "rgb_sim":
                flat_size = 32 * 32
            else:
                raise ValueError(f"Unknown perturbation mode: {perturbation_mode}")
                
            # Replace selected indices
            for pos in positions_to_change:
                if edge_guidance is not None:
                    # Use edge guidance for weighted sampling
                    flat_weights = edge_guidance.flatten()
                    # Exclude existing indices to avoid duplicates
                    mask = np.ones(flat_size, dtype=bool)
                    mask[new_indices] = False
                    available_weights = flat_weights * mask
                    if available_weights.sum() > 0:
                        available_weights = available_weights / available_weights.sum()
                        new_indices[pos] = np.random.choice(flat_size, p=available_weights)
                    else:
                        # Fallback to random if all positions taken
                        available_indices = np.setdiff1d(np.arange(flat_size), new_indices)
                        if len(available_indices) > 0:
                            new_indices[pos] = np.random.choice(available_indices)
                else:
                    # Uniform random sampling
                    available_indices = np.setdiff1d(np.arange(flat_size), new_indices)
                    if len(available_indices) > 0:
                        new_indices[pos] = np.random.choice(available_indices)
                        
        # Mutate perturbations
        for i in range(len(new_perturbations)):
            if np.random.rand() < mutation_prob:
                if self.config.continuous_perturbation.enable:
                    # Continuous perturbation mutation
                    if np.random.rand() < 0.9:  # 90% chance to slightly modify existing value
                        # Add small random change
                        change = np.random.normal(0, 0.1)
                        new_perturbations[i] = np.clip(
                            new_perturbations[i] + change,
                            self.config.continuous_perturbation.lower_bound,
                            self.config.continuous_perturbation.upper_bound
                        )
                        new_perturbations[i] = np.round(
                            new_perturbations[i], 
                            self.config.continuous_perturbation.decimal_places
                        )
                    else:  # 10% chance to completely resample
                        new_perturbations[i] = np.random.uniform(
                            low=self.config.continuous_perturbation.lower_bound,
                            high=self.config.continuous_perturbation.upper_bound
                        )
                        new_perturbations[i] = np.round(
                            new_perturbations[i], 
                            self.config.continuous_perturbation.decimal_places
                        )
                else:
                    # Discrete perturbation mutation
                    if np.random.rand() < 0.9:  # 90% chance to flip sign or set to zero
                        if np.random.rand() < zero_sample_prob:
                            new_perturbations[i] = 0
                        else:
                            # Flip sign or change to other non-zero value
                            if new_perturbations[i] == 0:
                                new_perturbations[i] = np.random.choice([-1, 1])
                            else:
                                # Keep same magnitude but maybe flip sign
                                if np.random.rand() < 0.5:
                                    new_perturbations[i] = -new_perturbations[i]
                    else:  # 10% chance to completely resample
                        new_perturbations[i] = np.random.choice([-1, 0, 1])
                        
        # Ensure at least some non-zero values to avoid trivial solutions
        if np.all(new_perturbations == 0):
            # Randomly select one position to be non-zero
            idx = np.random.randint(0, len(new_perturbations))
            if self.config.continuous_perturbation.enable:
                new_perturbations[idx] = np.random.uniform(
                    low=self.config.continuous_perturbation.lower_bound,
                    high=self.config.continuous_perturbation.upper_bound
                )
                new_perturbations[idx] = np.round(
                    new_perturbations[idx], 
                    self.config.continuous_perturbation.decimal_places
                )
            else:
                new_perturbations[idx] = np.random.choice([-1, 1])
                
        # Sort indices to maintain consistency
        sorted_idx = np.argsort(new_indices)
        new_indices = new_indices[sorted_idx]
        new_perturbations = new_perturbations[sorted_idx]
        
        return (new_indices, new_perturbations)
        
    def selection(
        self, 
        combined_population: List[Tuple[np.ndarray, np.ndarray]], 
        objective_values: List[Tuple[bool, float, float, float]], 
        pop_size: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Select the next generation population using non-dominated sorting
        
        Args:
            combined_population: Combined parent and offspring population
            objective_values: Objective values for each individual
            pop_size: Target population size
            
        Returns:
            Selected population for next generation
        """
        # Perform non-dominated sorting
        fronts = self._non_dominated_sort(objective_values)
        
        selected_pop = []
        remaining_slots = pop_size
        
        for front in fronts:
            if len(selected_pop) + len(front) <= pop_size:
                # Add entire front if it fits
                for idx in front:
                    selected_pop.append(combined_population[idx])
            else:
                # Need to partially fill from this front using crowding distance
                remaining_count = pop_size - len(selected_pop)
                if remaining_count > 0:
                    # Calculate crowding distances for individuals in this front
                    crowd_distances = self._calculate_crowding_distance(front, objective_values)
                    # Select individuals with largest crowding distances
                    sorted_by_distance = sorted(range(len(front)), key=lambda i: crowd_distances[i], reverse=True)
                    for i in range(min(remaining_count, len(sorted_by_distance))):
                        selected_pop.append(combined_population[front[sorted_by_distance[i]]])
                break
                
        return selected_pop
        
    def _non_dominated_sort(self, objective_values: List[Tuple[bool, float, float, float]]) -> List[List[int]]:
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
                    if self._dominates(objective_values[i], objective_values[j]):
                        S[i].append(j)
                    elif self._dominates(objective_values[j], objective_values[i]):
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
        
    def _dominates(self, obj1: Tuple[bool, float, float, float], obj2: Tuple[bool, float, float, float]) -> bool:
        """
        Check if objective tuple obj1 dominates obj2
        
        Args:
            obj1: First objective tuple (adversarial, loss, l2_norm, l0_norm)
            obj2: Second objective tuple (adversarial, loss, l2_norm, l0_norm)
            
        Returns:
            True if obj1 dominates obj2
        """
        # Check dominance according to the rules defined in the config
        # For simplicity, we'll implement a basic version here
        # A solution dominates another if it's better in at least one objective and not worse in others
        
        # First check adversarial status
        if obj1[0] and not obj2[0]:
            return True
        elif not obj1[0] and obj2[0]:
            return False
        # If both are adversarial or both are not, compare other objectives
        else:
            # Compare L2 norm (minimize)
            l2_better = obj1[2] < obj2[2]
            # Compare L0 norm (minimize)
            l0_better = obj1[3] < obj2[3]
            # Compare loss (minimize)
            loss_better = obj1[1] < obj2[1]
            
            # At least one objective must be better, and none worse
            at_least_one_better = l2_better or l0_better or loss_better
            none_worse = (not l2_better or obj1[2] <= obj2[2]) and \
                         (not l0_better or obj1[3] <= obj2[3]) and \
                         (not loss_better or obj1[1] <= obj2[1])
            
            return at_least_one_better and none_worse
            
    def _calculate_crowding_distance(self, front_indices: List[int], objective_values: List[Tuple[bool, float, float, float]]) -> List[float]:
        """
        Calculate crowding distance for individuals in a front
        
        Args:
            front_indices: Indices of individuals in the front
            objective_values: Objective values for all individuals
            
        Returns:
            Crowding distances for individuals in the front
        """
        distances = [0.0] * len(front_indices)
        
        if len(front_indices) <= 2:
            # Assign infinite distance to boundary solutions
            for i in range(len(front_indices)):
                distances[i] = float('inf')
            return distances
            
        # Get the subset of objective values for this front
        front_objs = [objective_values[i] for i in front_indices]
        
        # For each objective, sort the front and calculate distances
        # We'll consider L2 norm, L0 norm, and loss
        objectives = [[obj[1], obj[2], obj[3]] for obj in front_objs]  # [loss, l2_norm, l0_norm]
        
        for m in range(3):  # Three objectives
            # Create list of (index_in_front, objective_value)
            sorted_indices = sorted(range(len(front_objs)), key=lambda i: objectives[i][m])
            
            # Boundary points get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate distances for intermediate points
            if len(front_indices) > 2:
                # Get the range of this objective
                min_val = objectives[sorted_indices[0]][m]
                max_val = objectives[sorted_indices[-1]][m]
                range_val = max_val - min_val
                
                if range_val > 0:
                    for i in range(1, len(front_indices) - 1):
                        idx = sorted_indices[i]
                        prev_obj = objectives[sorted_indices[i-1]][m]
                        next_obj = objectives[sorted_indices[i+1]][m]
                        distances[idx] += (next_obj - prev_obj) / range_val
                        
        return distances