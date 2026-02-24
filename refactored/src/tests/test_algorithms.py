"""
Unit tests for the algorithms module
"""

import unittest
import numpy as np
from ..config.config_manager import ConfigManager
from ..algorithms.evolutionary_operators import EvolutionaryOperators
from ..algorithms.objective_functions import ObjectiveFunctions


class TestEvolutionaryOperators(unittest.TestCase):
    """Test cases for evolutionary operators"""
    
    def setUp(self):
        """Set up test fixtures"""
        config_manager = ConfigManager()
        self.operators = EvolutionaryOperators(config_manager.system_config)
        
    def test_initialize_population(self):
        """Test population initialization"""
        pop_size = 5
        fixed_k = 10
        perturbation_mode = "rgb_sim"
        
        population = self.operators.initialize_population(
            pop_size=pop_size,
            fixed_k=fixed_k,
            perturbation_mode=perturbation_mode
        )
        
        self.assertEqual(len(population), pop_size)
        
        for individual in population:
            indices, perturbations = individual
            self.assertEqual(len(indices), fixed_k)
            self.assertEqual(len(perturbations), fixed_k)
            self.assertTrue(np.all(indices >= 0))
            self.assertTrue(np.all(indices < 32*32))  # For rgb_sim mode
            # Check that indices are unique
            self.assertEqual(len(set(indices)), len(indices))
            
    def test_crossover(self):
        """Test crossover operation"""
        parent1 = (np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]))
        parent2 = (np.array([2, 3, 4]), np.array([0.4, 0.5, 0.6]))
        
        offspring1, offspring2 = self.operators.crossover(
            parent1, parent2, 
            fixed_k=3, 
            perturbation_mode="rgb_sim",
            crossover_prob=0.5
        )
        
        # Offspring should have the right size
        self.assertEqual(len(offspring1[0]), 3)
        self.assertEqual(len(offspring1[1]), 3)
        self.assertEqual(len(offspring2[0]), 3)
        self.assertEqual(len(offspring2[1]), 3)
        
    def test_mutation(self):
        """Test mutation operation"""
        individual = (np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]))
        
        mutated = self.operators.mutate(
            individual,
            fixed_k=3,
            mutation_prob=0.5,
            perturbation_mode="rgb_sim"
        )
        
        # Result should have same structure
        self.assertEqual(len(mutated[0]), 3)
        self.assertEqual(len(mutated[1]), 3)


class TestObjectiveFunctions(unittest.TestCase):
    """Test cases for objective functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        config_manager = ConfigManager()
        self.objectives = ObjectiveFunctions(config_manager.system_config)
        
    def test_compute_l2_norm(self):
        """Test L2 norm computation"""
        perturbations = np.array([3.0, 4.0])
        l2_norm = self.objectives._compute_l2_norm(perturbations)
        expected = 5.0  # sqrt(3^2 + 4^2)
        self.assertAlmostEqual(l2_norm, expected)
        
    def test_compute_l0_norm(self):
        """Test L0 norm computation"""
        perturbations = np.array([0, 1, 0, 2, 0, 3])
        l0_norm = self.objectives._compute_l0_norm(perturbations)
        expected = 3  # Count of non-zero elements
        self.assertEqual(l0_norm, expected)
        
    def test_dominates(self):
        """Test dominance relation"""
        # obj1: adversarial with better L2 and L0
        obj1 = (True, -2.0, 0.5, 5)
        # obj2: adversarial but worse L2 and L0
        obj2 = (True, -2.0, 1.0, 10)
        
        # obj1 should dominate obj2 since both are adversarial but obj1 has better L2 and L0
        self.assertTrue(self.objectives.dominates(obj1, obj2))
        
        # obj3: not adversarial
        obj3 = (False, -1.0, 0.5, 5)
        # obj4: adversarial
        obj4 = (True, -2.0, 1.0, 10)
        
        # obj4 should dominate obj3 since it's adversarial and obj3 is not
        self.assertTrue(self.objectives.dominates(obj4, obj3))


if __name__ == '__main__':
    unittest.main()