"""
Configuration types and data structures for SA-MOO
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model-related configuration"""
    weights_path: Path
    data_root_dir: Path
    output_dir: Path
    torch_home: Optional[Path] = None
    lpips_cache_dir: Optional[Path] = None


@dataclass
class SaMooConfig:
    """Core SA-MOO algorithm configuration"""
    population_size: int
    num_generations: int
    fixed_k: int
    crossover_prob: float
    zero_sample_prob: float


@dataclass
class AttackConfig:
    """Attack-specific configuration"""
    perturbation_mode: str  # 'rgb_sim', 'channel', 'v_channel'
    is_targeted_attack: bool
    target_class_id: Optional[int] = None


@dataclass
class ContinuousPerturbationConfig:
    """Continuous perturbation configuration"""
    enable: bool
    lower_bound: float
    upper_bound: float
    decimal_places: int


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    enable: bool
    interval: int


@dataclass
class RealWorldRobustnessConfig:
    """Real world robustness configuration"""
    enable: bool
    jpeg_quality: int
    enable_resize_preprocessing: bool
    resize_scale: float


@dataclass
class DominanceConfig:
    """Dynamic dominance relationship configuration"""
    rules: List[Dict[str, Any]]
    tie_breakers: List[Dict[str, Any]]


@dataclass
class DynamicParameterConfig:
    """Dynamic parameter scheduling configuration"""
    mutation_probability: Dict[str, Any]
    crossover_prob: Dict[str, Any]
    zero_sample_prob: Dict[str, Any]


@dataclass
class EdgeGuidanceConfig:
    """Edge guidance configuration"""
    enabled: bool
    method: str
    gaussian_sigma: float
    canny_sigma: float
    exponent: float
    uniform_mix: float
    min_value: float
    multi_scale: Dict[str, Any]
    semantic: Dict[str, Any]
    professional_preprocessing: Dict[str, Any]


@dataclass
class SystemConfig:
    """Complete system configuration"""
    target_image_id: int
    model: ModelConfig
    sa_moo: SaMooConfig
    attack: AttackConfig
    continuous_perturbation: ContinuousPerturbationConfig
    visualization: VisualizationConfig
    real_world_robustness: RealWorldRobustnessConfig
    dominance: DominanceConfig
    dynamic_parameters: DynamicParameterConfig
    edge_guidance: EdgeGuidanceConfig