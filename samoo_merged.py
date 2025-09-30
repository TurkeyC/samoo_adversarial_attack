#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SA-MOO Adversarial Attack - Merged Implementation
=================================================

This file contains all the Python code from the SA-MOO adversarial attack project 
merged into a single file, while preserving the original .env and config.yaml 
functionality.

Original project structure:
- src/config/config.py - Configuration management
- src/utils/logger.py - Logging utilities
- src/utils/edge_guidance.py - Edge guidance utilities
- src/data/data_loader.py - Data loading and model initialization
- src/core/objectives.py - Objective functions and dominance relations
- src/core/evolutionary_operators.py - Evolutionary operators
- src/core/dynamic_parameters.py - Dynamic parameter scheduling
- src/visualization/visualization.py - Visualization and result saving
- src/main.py - Main execution module

All dependencies on .env files and config.yaml files are preserved.
"""

# Standard library imports
import os
import sys
import json
import copy
import argparse
import warnings
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Sequence
from dataclasses import dataclass
from io import BytesIO

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18
import lpips
import yaml
from dotenv import load_dotenv
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")  # Use interactive plotting backend
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# Scikit-image imports
from skimage.color import rgb2lab, deltaE_ciede2000, rgb2gray
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.filters import gaussian, sobel, scharr, prewitt
from skimage.feature import canny
from skimage.io import imread
from skimage.transform import resize

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION MANAGEMENT MODULE
# =============================================================================

class Config:
    """
    Configuration management class that supports multiple configuration sources.

    Priority order (from high to low):
    1. Command line arguments
    2. Environment variables
    3. YAML configuration file
    4. Default values
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_file: YAML configuration file path, if None uses default path
        """
        self.config_file = config_file or self._find_config_file()
        self.yaml_config = self._load_yaml_config()
        self.args = self._parse_args()

        # Initialize all configuration parameters
        self._init_config()

    def _find_config_file(self) -> Optional[str]:
        """Find configuration file"""
        # First try environment variable specified config file
        config_path = os.getenv("CONFIG_FILE")
        if config_path and Path(config_path).exists():
            return config_path

        # Search for default config files
        default_configs = ["config.yaml", "config.yml", "../config.yaml", "../config.yml"]
        for config_name in default_configs:
            config_path = Path(__file__).parent / config_name
            if config_path.exists():
                return str(config_path)

        return None

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not self.config_file or not Path(self.config_file).exists():
            return {}

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load config file {self.config_file}: {e}")
            return {}

    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description="SA-MOO Adversarial Attack")

        # Core configuration
        parser.add_argument("--target-image-id", type=int, help="CIFAR-10 test set image index")
        parser.add_argument("--model-weights-path", type=str, help="Model weights file path")
        parser.add_argument("--data-root-dir", type=str, help="CIFAR-10 data root directory")
        parser.add_argument("--output-dir", type=str, help="Output directory")

        # SA-MOO hyperparameters
        parser.add_argument("--population-size", type=int, help="Population size")
        parser.add_argument("--num-generations", type=int, help="Maximum generations")
        parser.add_argument("--fixed-k", type=int, help="Perturbation budget K")
        parser.add_argument("--crossover-prob", type=float, help="Crossover probability")
        parser.add_argument("--zero-sample-prob", type=float, help="Zero sampling probability")

        # Attack configuration
        parser.add_argument("--perturbation-mode", type=str,
                          choices=["rgb_sim", "channel", "v_channel"], help="Perturbation mode")
        parser.add_argument("--targeted", action="store_true", help="Whether to use targeted attack")
        parser.add_argument("--target-class-id", type=int, help="Target class ID")

        # Continuous perturbation configuration
        parser.add_argument("--enable-continuous-perturbation", action="store_true", help="Enable continuous perturbation")
        parser.add_argument("--continuous-lower-bound", type=float, help="Continuous perturbation lower bound")
        parser.add_argument("--continuous-upper-bound", type=float, help="Continuous perturbation upper bound")
        parser.add_argument("--continuous-decimal-places", type=int, help="Continuous perturbation decimal places")

        # Visualization configuration
        parser.add_argument("--enable-visualization", action="store_true", help="Enable visualization")
        parser.add_argument("--visualize-interval", type=int, help="Visualization interval")

        # Dynamic configuration
        parser.add_argument("--dominance-config", type=str, help="Dynamic dominance configuration (JSON string or file path)")
        parser.add_argument("--dynamic-params", type=str, help="Dynamic parameter configuration (JSON string or file path)")
        parser.add_argument("--edge-guidance", type=str, help="Edge guidance configuration (JSON string or file path)")

        # Other
        parser.add_argument("--config", type=str, help="Specify configuration file path")

        return parser.parse_args()

    def _get_config_value(self, key: str, default_value: Any, value_type: type = str) -> Any:
        """
        Get configuration value from multiple sources by priority

        Args:
            key: Configuration key name
            default_value: Default value
            value_type: Value type

        Returns:
            Configuration value
        """
        # 1. Command line arguments have highest priority
        arg_key = key.replace('_', '-')
        if hasattr(self.args, arg_key) and getattr(self.args, arg_key) is not None:
            return getattr(self.args, arg_key)

        # 2. Environment variables
        env_key = f"SA_MOO_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            if value_type == bool:
                return env_value.lower() in ('true', '1', 'yes', 'on')
            return value_type(env_value)

        # 3. YAML configuration file
        if key in self.yaml_config:
            value = self.yaml_config[key]
            if value_type == bool and isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return value_type(value) if value is not None else default_value

        # 4. Default value
        return default_value

    def _parse_external_config(self, raw_value: Optional[str], description: str) -> Dict[str, Any]:
        """Parse external configuration, can be JSON string or file path."""
        if not raw_value:
            return {}

        candidate_path = Path(raw_value)
        if candidate_path.exists():
            try:
                with open(candidate_path, "r", encoding="utf-8") as f:
                    if candidate_path.suffix.lower() in {".yml", ".yaml"}:
                        return yaml.safe_load(f) or {}
                    return json.load(f)
            except Exception as exc:
                raise ValueError(f"Failed to load {description} from file '{raw_value}': {exc}") from exc

        try:
            return json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {description} JSON string: {exc}") from exc

    @staticmethod
    def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries without modifying input dictionaries."""
        if not overrides:
            return base

        result = copy.deepcopy(base)
        stack: List[tuple] = [(result, overrides)]
        while stack:
            current_base, current_override = stack.pop()
            for key, value in current_override.items():
                if isinstance(value, dict) and isinstance(current_base.get(key), dict):
                    stack.append((current_base[key], value))
                else:
                    current_base[key] = copy.deepcopy(value)
        return result

    def _load_dominance_config(self) -> Dict[str, Any]:
        """Load dynamic dominance relationship configuration."""
        default_config: Dict[str, Any] = {
            "rules": [
                {
                    "when": {"is_adversarial": True, "other_is_adversarial": False},
                    "prefer": "self"
                },
                {
                    "when": {"is_adversarial": False, "other_is_adversarial": True},
                    "prefer": "other"
                },
                {
                    "when": {"is_adversarial": True, "other_is_adversarial": True},
                    "metrics": [
                        {"name": "l2_norm", "goal": "min", "tolerance": 0.0},
                        {"name": "l0_norm", "goal": "min", "tolerance": 0.0}
                    ]
                },
                {
                    "when": {"is_adversarial": False, "other_is_adversarial": False},
                    "metrics": [
                        {"name": "loss", "goal": "min", "tolerance": 0.0},
                        {"name": "l2_norm", "goal": "min", "tolerance": 0.0}
                    ]
                }
            ],
            "tie_breakers": [
                {"name": "l0_norm", "goal": "min", "tolerance": 0.0}
            ]
        }

        dominance_config = copy.deepcopy(default_config)

        yaml_override = self.yaml_config.get("dominance") if isinstance(self.yaml_config, dict) else None
        if isinstance(yaml_override, dict):
            dominance_config = self._deep_update(dominance_config, yaml_override)

        env_override = os.getenv("SA_MOO_DOMINANCE_CONFIG")
        if env_override:
            dominance_config = self._deep_update(dominance_config, self._parse_external_config(env_override, "dominance config"))

        arg_override = getattr(self.args, "dominance_config", None)
        if arg_override:
            dominance_config = self._deep_update(dominance_config, self._parse_external_config(arg_override, "dominance config"))

        return dominance_config

    def _load_dynamic_parameters(self) -> Dict[str, Any]:
        """Load dynamic parameter adjustment configuration."""
        default_config: Dict[str, Any] = {
            "mutation_probability": {
                "schedule": "linear",
                "targeted": {"start": 0.2, "end": 0.01},
                "non_targeted": {"start": 0.4, "end": 0.01},
                "min": 0.0,
                "max": 1.0
            },
            "crossover_prob": {
                "schedule": "constant",
                "value": self.CROSSOVER_PROB
            },
            "zero_sample_prob": {
                "schedule": "constant",
                "value": self.ZERO_SAMPLE_PROB
            }
        }

        dynamic_config = copy.deepcopy(default_config)

        yaml_override = self.yaml_config.get("dynamic_parameters") if isinstance(self.yaml_config, dict) else None
        if isinstance(yaml_override, dict):
            dynamic_config = self._deep_update(dynamic_config, yaml_override)

        env_override = os.getenv("SA_MOO_DYNAMIC_PARAMS")
        if env_override:
            dynamic_config = self._deep_update(dynamic_config, self._parse_external_config(env_override, "dynamic parameter config"))

        arg_override = getattr(self.args, "dynamic_params", None)
        if arg_override:
            dynamic_config = self._deep_update(dynamic_config, self._parse_external_config(arg_override, "dynamic parameter config"))

        return dynamic_config

    def _load_edge_guidance_config(self) -> Dict[str, Any]:
        """Load edge guidance configuration."""
        default_config: Dict[str, Any] = {
            "enabled": False,
            "method": "sobel",
            "gaussian_sigma": 0.8,
            "canny_sigma": 1.0,
            "exponent": 1.5,
            "uniform_mix": 0.15,
            "min_value": 1e-4,
            "multi_scale": {
                "enabled": False,
                "scales": [1.0, 0.5, 0.25],
                "weights": None,
                "gaussian_sigmas": None,
                "combine": "mean",
                "anti_aliasing": True,
            },
            "semantic": {
                "enabled": False,
                "path": None,
                "array": None,
                "weight": 0.5,
                "blend_mode": "multiply",
                "normalize": True,
                "invert": False,
                "blur_sigma": 0.0,
                "exponent": 1.0,
                "clip_low": None,
                "clip_high": None,
                "fail_on_missing": False,
            },
            "professional_preprocessing": {
                "enabled": False,
                "edge_map_path": None,
                "edge_normalize": False,
                "edge_clip_low": None,
                "edge_clip_high": None,
                "semantic_map_path": None,
                "semantic_normalize": True,
                "semantic_clip_low": None,
                "semantic_clip_high": None,
                "fail_on_missing": False,
                "semantic_model": {
                    "enabled": False,
                    "name": "deeplabv3_resnet50",
                    "device": "auto",
                    "output": "max_prob",
                    "target_class": None,
                    "smooth_sigma": 0.0,
                    "entropy_eps": 1e-6,
                },
                "edge_model": {
                    "enabled": False,
                    "type": "semantic_gradient",
                    "device": "auto",
                    "normalize": True,
                    "smooth_sigma": 0.0,
                    "canny_sigma": 1.0,
                    "semantic_output": "max_prob",
                    "semantic_smooth_sigma": 0.0,
                    "entropy_eps": 1e-6,
                    "target_class": None,
                },
            },
        }

        edge_config = copy.deepcopy(default_config)

        yaml_override = self.yaml_config.get("edge_guidance") if isinstance(self.yaml_config, dict) else None
        if isinstance(yaml_override, dict):
            edge_config = self._deep_update(edge_config, yaml_override)

        env_override = os.getenv("SA_MOO_EDGE_GUIDANCE")
        if env_override:
            edge_config = self._deep_update(edge_config, self._parse_external_config(env_override, "edge guidance config"))

        arg_override = getattr(self.args, "edge_guidance", None)
        if arg_override:
            edge_config = self._deep_update(edge_config, self._parse_external_config(arg_override, "edge guidance config"))

        return edge_config

    def _init_config(self):
        """Initialize all configuration parameters"""
        # ==================== Core configuration ====================
        self.TARGET_IMAGE_ID: int = self._get_config_value("target_image_id", 7780, int)

        # Model weights path handling - prioritize reading from environment variables
        weights_path = os.getenv("MODEL_WEIGHTS_PATH") or self._get_config_value("model_weights_path", "cifar10_resnet18.pth", str)
        self.MODEL_WEIGHTS_PATH: Path = Path(weights_path)
        if not self.MODEL_WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Model weights not found at {self.MODEL_WEIGHTS_PATH.absolute()}.")

        # Data and output directories - prioritize reading from environment variables
        data_root = os.getenv("DATA_ROOT_DIR") or self._get_config_value("data_root_dir", "./data", str)
        self.DATA_ROOT_DIR: Path = Path(data_root)

        output_dir = os.getenv("OUTPUT_DIR") or self._get_config_value("output_dir", "../output", str)
        self.OUTPUT_DIR: Path = Path(output_dir)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Torch environment variables
        torch_home = os.getenv("TORCH_HOME")
        lpips_cache = os.getenv("LPIPS_CACHE_DIR")
        if torch_home:
            self.TORCH_HOME: Path = Path(torch_home).resolve()
        else:
            self.TORCH_HOME = Path.home() / ".cache" / "torch"
        if lpips_cache:
            self.LPIPS_CACHE_DIR: Path = Path(lpips_cache).resolve()
        else:
            self.LPIPS_CACHE_DIR = Path.home() / ".cache" / "torch" / "lpips"

        os.environ["TORCH_HOME"] = str(self.TORCH_HOME)
        os.environ["LPIPS_CACHE_DIR"] = str(self.LPIPS_CACHE_DIR)

        # ==================== SA-MOO hyperparameters ====================
        self.POPULATION_SIZE: int = self._get_config_value("population_size", 2, int)
        self.NUM_GENERATIONS: int = self._get_config_value("num_generations", 1000, int)
        self.FIXED_K: int = self._get_config_value("fixed_k", 24, int)
        self.CROSSOVER_PROB: float = self._get_config_value("crossover_prob", 0.1, float)
        self.ZERO_SAMPLE_PROB: float = self._get_config_value("zero_sample_prob", 0.3, float)

        # ==================== Attack configuration ====================
        self.PERTURBATION_MODE: str = self._get_config_value("perturbation_mode", "rgb_sim", str)
        self.IS_TARGETED_ATTACK: bool = self._get_config_value("is_targeted_attack", False, bool)
        self.TARGET_CLASS_ID: Optional[int] = self._get_config_value("target_class_id", 5, int)

        # ==================== Continuous perturbation configuration ====================
        self.ENABLE_CONTINUOUS_PERTURBATION: bool = self._get_config_value("enable_continuous_perturbation", False, bool)
        self.CONTINUOUS_LOWER_BOUND: float = self._get_config_value("continuous_lower_bound", -1.0, float)
        self.CONTINUOUS_UPPER_BOUND: float = self._get_config_value("continuous_upper_bound", 1.0, float)
        self.CONTINUOUS_DECIMAL_PLACES: int = self._get_config_value("continuous_decimal_places", 2, int)

        # ==================== Visualization configuration ====================
        self.ENABLE_VISUALIZATION: bool = self._get_config_value("enable_visualization", False, bool)
        self.VISUALIZE_INTERVAL: int = self._get_config_value("visualize_interval", 500, int)

        # ==================== Real world robustness configuration ====================
        self.ENABLE_REAL_WORLD_ROBUSTNESS: bool = self._get_config_value("enable_real_world_robustness", False, bool)
        self.JPEG_QUALITY: int = self._get_config_value("jpeg_quality", 75, int)
        self.ENABLE_RESIZE_PREPROCESSING: bool = self._get_config_value("enable_resize_preprocessing", True, bool)
        self.RESIZE_SCALE: float = self._get_config_value("resize_scale", 2.0, float)

        # ==================== Dynamic configuration ====================
        self.DOMINANCE_CONFIG: Dict[str, Any] = self._load_dominance_config()
        self.DYNAMIC_PARAMETER_CONFIG: Dict[str, Any] = self._load_dynamic_parameters()
        self.EDGE_GUIDANCE_CONFIG: Dict[str, Any] = self._load_edge_guidance_config()

    def print_config(self):
        """Print current configuration"""
        print("=== Current Configuration ===")
        print(f"Target Image ID: {self.TARGET_IMAGE_ID}")
        print(f"Model Weights: {self.MODEL_WEIGHTS_PATH}")
        print(f"Data Root: {self.DATA_ROOT_DIR}")
        print(f"Output Dir: {self.OUTPUT_DIR}")
        print(f"Perturbation Mode: {self.PERTURBATION_MODE}")
        print(f"Targeted Attack: {self.IS_TARGETED_ATTACK}")
        if self.IS_TARGETED_ATTACK:
            print(f"Target Class ID: {self.TARGET_CLASS_ID}")
        print(f"Continuous Perturbation: {self.ENABLE_CONTINUOUS_PERTURBATION}")
        if self.ENABLE_CONTINUOUS_PERTURBATION:
            print(f"Continuous Bounds: [{self.CONTINUOUS_LOWER_BOUND}, {self.CONTINUOUS_UPPER_BOUND}]")
            print(f"Decimal Places: {self.CONTINUOUS_DECIMAL_PLACES}")
        print(f"Population Size: {self.POPULATION_SIZE}")
        print(f"Generations: {self.NUM_GENERATIONS}")
        print(f"Fixed K: {self.FIXED_K}")
        print(f"Visualization: {self.ENABLE_VISUALIZATION}")
        print(f"Real World Robustness: {self.ENABLE_REAL_WORLD_ROBUSTNESS}")
        print("=" * 30)


# Create global configuration instance
config = Config()

# For backward compatibility, export configuration values as module-level variables
TARGET_IMAGE_ID = config.TARGET_IMAGE_ID
MODEL_WEIGHTS_PATH = config.MODEL_WEIGHTS_PATH
DATA_ROOT_DIR = config.DATA_ROOT_DIR
OUTPUT_DIR = config.OUTPUT_DIR
TORCH_HOME = config.TORCH_HOME
LPIPS_CACHE_DIR = config.LPIPS_CACHE_DIR
POPULATION_SIZE = config.POPULATION_SIZE
NUM_GENERATIONS = config.NUM_GENERATIONS
FIXED_K = config.FIXED_K
CROSSOVER_PROB = config.CROSSOVER_PROB
ZERO_SAMPLE_PROB = config.ZERO_SAMPLE_PROB
PERTURBATION_MODE = config.PERTURBATION_MODE
IS_TARGETED_ATTACK = config.IS_TARGETED_ATTACK
TARGET_CLASS_ID = config.TARGET_CLASS_ID
ENABLE_CONTINUOUS_PERTURBATION = config.ENABLE_CONTINUOUS_PERTURBATION
CONTINUOUS_LOWER_BOUND = config.CONTINUOUS_LOWER_BOUND
CONTINUOUS_UPPER_BOUND = config.CONTINUOUS_UPPER_BOUND
CONTINUOUS_DECIMAL_PLACES = config.CONTINUOUS_DECIMAL_PLACES
ENABLE_VISUALIZATION = config.ENABLE_VISUALIZATION
VISUALIZE_INTERVAL = config.VISUALIZE_INTERVAL
ENABLE_REAL_WORLD_ROBUSTNESS = config.ENABLE_REAL_WORLD_ROBUSTNESS
JPEG_QUALITY = config.JPEG_QUALITY
ENABLE_RESIZE_PREPROCESSING = config.ENABLE_RESIZE_PREPROCESSING
RESIZE_SCALE = config.RESIZE_SCALE
DOMINANCE_CONFIG = config.DOMINANCE_CONFIG
DYNAMIC_PARAMETER_CONFIG = config.DYNAMIC_PARAMETER_CONFIG
EDGE_GUIDANCE_CONFIG = config.EDGE_GUIDANCE_CONFIG


# =============================================================================
# LOGGER MODULE
# =============================================================================

class Logger:
    """
    Custom logger class that outputs to both console and file.

    This class redirects sys.stdout to make all print statements output to both
    console and specified log file. Suitable for scenarios requiring both
    real-time output viewing and complete log saving.

    Attributes:
        terminal: Original sys.stdout for console output
        log_file: Opened log file object for file output
    """

    def __init__(self, filepath: Path):
        """
        Initialize Logger instance.

        Args:
            filepath: Complete path to log file. File will be created or overwritten.
        """
        self.terminal = sys.stdout  # Save original stdout
        self.log_file = open(filepath, "w", encoding="utf-8")  # Open log file in write mode

    def write(self, message: str):
        """
        Write message to both console and log file.

        This method is called by sys.stdout redirection to achieve simultaneous output.

        Args:
            message: Message string to write
        """
        self.terminal.write(message)  # Output to console
        self.log_file.write(message)  # Output to file

    def flush(self):
        """
        Flush output buffers.

        Ensure messages are written immediately to console and file rather than waiting for buffer to fill.
        """
        self.terminal.flush()  # Flush console buffer
        self.log_file.flush()  # Flush file buffer

    def close(self):
        """
        Close log file.

        Call this method at program end to ensure all data is written to file.
        """
        self.log_file.close()  # Close file object


# =============================================================================
# EDGE GUIDANCE MODULE
# =============================================================================

_TORCHVISION_SEGMENTATION_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


@dataclass
class EdgeGuidanceWeights:
    """Container for per-domain sampling probabilities.

    Attributes:
        pixel_weights: Flattened probability distribution over image pixels.
            The length should be H * W.
        channel_weights: Flattened probability distribution over individual
            RGB channels. The length should be H * W * 3 and must sum to 1.
        artifacts: Optional debug artifacts (e.g., edge maps, semantic maps)
            captured during weight construction for visualization purposes.
    """

    pixel_weights: np.ndarray
    channel_weights: np.ndarray
    artifacts: Optional[Dict[str, np.ndarray]] = None

    def weights_for_mode(self, mode: str) -> np.ndarray:
        """Return the appropriate weight vector for the given perturbation mode."""
        if mode == "channel":
            return self.channel_weights
        return self.pixel_weights

    def sample_without_replacement(
        self,
        mode: str,
        candidate_indices: np.ndarray,
        sample_size: int,
    ) -> np.ndarray:
        """Sample indices with guidance, falling back to uniform if needed."""
        if sample_size <= 0 or candidate_indices.size == 0:
            return np.empty(0, dtype=int)

        weights = self.weights_for_mode(mode)
        subset_weights = weights[candidate_indices]
        total = float(subset_weights.sum())

        sample_size = min(sample_size, candidate_indices.size)
        if not np.isfinite(total) or total <= 0.0:
            return np.random.choice(candidate_indices, size=sample_size, replace=False)

        probabilities = subset_weights / total
        if np.any(~np.isfinite(probabilities)):
            return np.random.choice(candidate_indices, size=sample_size, replace=False)

        # Guard against numerical precision issues that might create slight negatives
        probabilities = np.clip(probabilities, 0.0, None)
        prob_sum = probabilities.sum()
        if prob_sum <= 0.0:
            return np.random.choice(candidate_indices, size=sample_size, replace=False)

        probabilities /= prob_sum
        return np.random.choice(candidate_indices, size=sample_size, replace=False, p=probabilities)


def _normalize_weights(flat_weights: np.ndarray, min_value: float, uniform_mix: float) -> np.ndarray:
    """Normalize and smooth a flattened weight map."""
    if flat_weights.size == 0:
        return np.array([], dtype=np.float64)

    flat = np.clip(flat_weights, 0.0, None)
    if not np.isfinite(flat).all():
        if flat.size == 0:
            return np.array([], dtype=np.float64)
        return np.full(flat.size, 1.0 / flat.size, dtype=np.float64)

    flat_min = float(flat.min(initial=0.0))
    flat -= flat_min
    flat_max = float(flat.max(initial=0.0))
    if flat_max > 0.0:
        flat /= flat_max

    flat += max(min_value, 0.0)
    total = float(flat.sum())
    if total <= 0.0 or not np.isfinite(total):
        flat = np.full(flat.size, 1.0 / max(flat.size, 1))
    else:
        flat /= total

    uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
    if uniform_mix > 0.0 and flat.size > 0:
        uniform = np.full(flat.size, 1.0 / flat.size)
        flat = (1.0 - uniform_mix) * flat + uniform_mix * uniform
        flat /= float(flat.sum())

    return flat.astype(np.float64)


def _compute_core_edges(gray: np.ndarray, method: str, canny_sigma: float) -> np.ndarray:
    """Compute edge magnitude using the requested detector."""
    if method == "sobel":
        edges = sobel(gray)
    elif method == "scharr":
        edges = scharr(gray)
    elif method == "prewitt":
        edges = prewitt(gray)
    elif method == "canny":
        edges = canny(gray, sigma=canny_sigma).astype(np.float32)
    else:
        raise ValueError(f"Unsupported edge guidance method: {method}")

    return np.abs(edges.astype(np.float64))


def _compute_multi_scale_edges(
    gray: np.ndarray,
    method: str,
    base_gaussian_sigma: float,
    canny_sigma: float,
    multi_cfg: Dict[str, object],
) -> np.ndarray:
    """Aggregate edges computed over multiple spatial scales."""

    scales = multi_cfg.get("scales", [1.0, 0.5, 0.25])
    if not isinstance(scales, Sequence) or isinstance(scales, (str, bytes)):
        scales = [1.0]
    scales = list(scales)
    if all(abs(float(scale) - 1.0) > 1e-6 for scale in scales):
        scales.insert(0, 1.0)

    weights = multi_cfg.get("weights")
    gaussian_sigmas = multi_cfg.get("gaussian_sigmas")
    combine_mode = str(multi_cfg.get("combine", "mean")).lower()
    anti_aliasing = bool(multi_cfg.get("anti_aliasing", True))

    height, width = gray.shape
    aggregated = None
    weight_sum = 0.0

    for idx, scale_value in enumerate(scales):
        try:
            scale = float(scale_value)
        except (TypeError, ValueError):
            continue
        if scale <= 0.0:
            continue

        if abs(scale - 1.0) < 1e-6:
            scaled_gray = gray
        else:
            new_shape = (
                max(1, int(round(height * scale))),
                max(1, int(round(width * scale))),
            )
            scaled_gray = resize(
                gray,
                new_shape,
                mode="reflect",
                anti_aliasing=anti_aliasing,
                preserve_range=True,
            )

        sigma = base_gaussian_sigma
        if isinstance(gaussian_sigmas, Sequence) and not isinstance(gaussian_sigmas, (str, bytes)):
            if idx < len(gaussian_sigmas):
                try:
                    sigma = float(gaussian_sigmas[idx])
                except (TypeError, ValueError):
                    sigma = base_gaussian_sigma

        if sigma > 0.0:
            scaled_gray = gaussian(scaled_gray, sigma=sigma, mode="reflect")

        scaled_edges = _compute_core_edges(scaled_gray, method, canny_sigma)

        if scaled_edges.shape != (height, width):
            scaled_edges = resize(
                scaled_edges,
                (height, width),
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            )

        if aggregated is None:
            aggregated = np.zeros_like(scaled_edges)

        if combine_mode == "max":
            aggregated = np.maximum(aggregated, scaled_edges)
            continue

        weight = 1.0
        if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
            if idx < len(weights):
                try:
                    weight = float(weights[idx])
                except (TypeError, ValueError):
                    weight = 1.0

        aggregated += weight * scaled_edges
        weight_sum += weight

    if aggregated is None:
        if base_gaussian_sigma > 0.0:
            blurred = gaussian(gray, sigma=base_gaussian_sigma, mode="reflect")
        else:
            blurred = gray
        return _compute_core_edges(blurred, method, canny_sigma)

    if combine_mode == "max":
        return aggregated

    if weight_sum > 0.0:
        aggregated = aggregated / weight_sum

    return aggregated


def _load_semantic_map(
    semantic_cfg: Dict[str, object],
    target_shape: Sequence[int],
) -> Optional[np.ndarray]:
    """Load and preprocess a semantic attention map if configured."""

    path_value = semantic_cfg.get("path")
    semantic_array = semantic_cfg.get("array")
    if path_value is None and semantic_array is None:
        return None

    data: Optional[np.ndarray] = None

    if path_value is not None:
        path = Path(str(path_value))
        if not path.exists():
            if semantic_cfg.get("fail_on_missing", False):
                raise FileNotFoundError(f"Semantic map not found at: {path}")
            return None

        suffix = path.suffix.lower()
        if suffix in {".npy", ".npz"}:
            loaded = np.load(path)
            if isinstance(loaded, np.ndarray):
                data = loaded
            else:
                # npz archive: take the first array
                first_key = next(iter(loaded.files))
                data = loaded[first_key]
        else:
            data = imread(path)

    if data is None and semantic_array is not None:
        data = np.asarray(semantic_array)

    if data is None:
        return None

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 3:
        if data.shape[2] == 3:
            data = rgb2gray(data)
        else:
            data = np.mean(data, axis=2)
    elif data.ndim == 1 and data.size == int(target_shape[0]) * int(target_shape[1]):
        data = data.reshape(tuple(target_shape[:2]))
    elif data.ndim != 2:
        raise ValueError("Semantic map must be 2D or broadcastable to the image shape")

    target_hw = (int(target_shape[0]), int(target_shape[1]))
    if data.shape != target_hw:
        data = resize(
            data,
            target_hw,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )

    if semantic_cfg.get("normalize", True):
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        if np.isfinite(data_max - data_min) and data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

    if semantic_cfg.get("invert", False):
        data = 1.0 - data

    exponent = float(semantic_cfg.get("exponent", 1.0))
    if exponent != 1.0:
        data = np.power(np.clip(data, 0.0, None), exponent)

    blur_sigma = float(semantic_cfg.get("blur_sigma", 0.0))
    if blur_sigma > 0.0:
        data = gaussian(data, sigma=blur_sigma, mode="reflect")

    clip_low = semantic_cfg.get("clip_low")
    clip_high = semantic_cfg.get("clip_high")
    if clip_low is not None or clip_high is not None:
        lo = float(clip_low) if clip_low is not None else 0.0
        hi = float(clip_high) if clip_high is not None else np.inf
        data = np.clip(data, lo, hi)

    return np.clip(data, 0.0, None)


def _load_external_map(
    path_value: Optional[object],
    target_shape: Sequence[int],
    *,
    normalize: bool = False,
    clip_low: Optional[float] = None,
    clip_high: Optional[float] = None,
    fail_on_missing: bool = False,
) -> Optional[np.ndarray]:
    """Load a generic map produced by an external professional model."""

    if path_value is None:
        return None

    path = Path(str(path_value))
    if not path.exists():
        if fail_on_missing:
            raise FileNotFoundError(f"Professional preprocessing map not found at: {path}")
        return None

    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        loaded = np.load(path)
        if isinstance(loaded, np.ndarray):
            data = loaded
        else:
            first_key = next(iter(loaded.files))
            data = loaded[first_key]
    else:
        data = imread(path)

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 3:
        if data.shape[2] == 3:
            data = rgb2gray(data)
        else:
            data = np.mean(data, axis=2)
    elif data.ndim == 1 and data.size == int(target_shape[0]) * int(target_shape[1]):
        data = data.reshape(tuple(target_shape[:2]))
    elif data.ndim != 2:
        raise ValueError("Professional preprocessing map must be 2D or broadcastable to the image shape")

    target_hw = (int(target_shape[0]), int(target_shape[1]))
    if data.shape != target_hw:
        data = resize(
            data,
            target_hw,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )

    if normalize:
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        if np.isfinite(data_max - data_min) and data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

    if clip_low is not None or clip_high is not None:
        lo = float(clip_low) if clip_low is not None else -np.inf
        hi = float(clip_high) if clip_high is not None else np.inf
        data = np.clip(data, lo, hi)

    return np.asarray(data, dtype=np.float64)


def _normalize_map(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return data
    data = np.clip(data, 0.0, None)
    minimum = float(np.min(data))
    maximum = float(np.max(data))
    if maximum > minimum:
        data = (data - minimum) / (maximum - minimum)
    else:
        data = np.zeros_like(data)
    return data


def _resolve_device(device_pref: str) -> str:
    desired = str(device_pref or "cpu").lower()
    if desired in {"auto", "best"}:
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return desired


def _get_torchvision_segmentation_model(name: str, device: str) -> Tuple[Any, Any]:
    cache_key = (name, device)
    if cache_key in _TORCHVISION_SEGMENTATION_CACHE:
        return _TORCHVISION_SEGMENTATION_CACHE[cache_key]

    try:
        import torch  # type: ignore
        from torchvision.models.segmentation import (  # type: ignore
            deeplabv3_resnet50,
            DeepLabV3_ResNet50_Weights,
            fcn_resnet50,
            FCN_ResNet50_Weights,
            lraspp_mobilenet_v3_large,
            LRASPP_MobileNet_V3_Large_Weights,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Torchvision (with segmentation models) is required for professional preprocessing."
        ) from exc

    device_obj = torch.device(device)

    if name == "deeplabv3_resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
    elif name == "fcn_resnet50":
        weights = FCN_ResNet50_Weights.DEFAULT
        model = fcn_resnet50(weights=weights)
    elif name == "lraspp_mobilenet_v3_large":
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        model = lraspp_mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unsupported torchvision segmentation model: {name}")

    preprocess = weights.transforms()
    model.eval()
    model.to(device_obj)

    _TORCHVISION_SEGMENTATION_CACHE[cache_key] = (model, preprocess)
    return model, preprocess


def _run_semantic_model_from_cfg(
    rgb_image: np.ndarray,
    semantic_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Running built-in semantic models requires torch and torchvision to be installed."
        ) from exc

    from PIL import Image  # Local import to avoid mandatory dependency when unused

    model_name = str(semantic_cfg.get("name", "deeplabv3_resnet50"))
    device = _resolve_device(str(semantic_cfg.get("device", "auto")))
    model, preprocess = _get_torchvision_segmentation_model(model_name, device)

    pil_image = Image.fromarray(
        np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    )
    processed = preprocess(pil_image)
    if processed.dim() == 3:
        processed = processed.unsqueeze(0)

    try:
        input_tensor = processed.to(device)
    except AttributeError:
        # Older torchvision transforms may return numpy arrays
        input_tensor = torch.from_numpy(processed).to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"]
        probabilities = torch.softmax(output, dim=1)
        probabilities = F.interpolate(
            probabilities,
            size=rgb_image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )

    output_mode = str(semantic_cfg.get("output", "max_prob")).lower()
    if output_mode == "max_prob":
        semantic_tensor = probabilities.max(dim=1).values.squeeze(0)
    elif output_mode == "entropy":
        eps = float(semantic_cfg.get("entropy_eps", 1e-6))
        semantic_tensor = -torch.sum(
            probabilities * torch.log(probabilities + eps), dim=1
        ).squeeze(0)
    elif output_mode == "target_class":
        target_idx = semantic_cfg.get("target_class")
        if target_idx is None:
            raise ValueError("semantic_model.target_class must be set when output='target_class'")
        target_idx = int(target_idx)
        if target_idx < 0 or target_idx >= probabilities.shape[1]:
            raise ValueError("semantic_model.target_class is out of range for the selected model")
        semantic_tensor = probabilities[:, target_idx, :, :].squeeze(0)
    elif output_mode == "top2_gap":
        top2 = torch.topk(probabilities, k=2, dim=1).values
        semantic_tensor = (top2[:, 0, :, :] - top2[:, 1, :, :]).squeeze(0)
    else:
        raise ValueError(f"Unsupported semantic_model.output value: {output_mode}")

    semantic_map = semantic_tensor.cpu().numpy()
    sigma = float(semantic_cfg.get("smooth_sigma", 0.0))
    if sigma > 0.0:
        semantic_map = gaussian(semantic_map, sigma=sigma, mode="reflect")

    return np.clip(semantic_map, 0.0, None), probabilities.squeeze(0).cpu().numpy()


def _generate_edge_map_from_cfg(
    rgb_image: np.ndarray,
    edge_cfg: Dict[str, Any],
    semantic_map: Optional[np.ndarray],
    semantic_probabilities: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    method = str(edge_cfg.get("type", "semantic_gradient")).lower()

    if method == "semantic_gradient":
        if semantic_map is None:
            raise ValueError("semantic_gradient edge model requires a semantic map.")
        grad_y, grad_x = np.gradient(semantic_map)
        edges = np.sqrt(grad_x ** 2 + grad_y ** 2)
    elif method == "semantic_entropy":
        if semantic_probabilities is None:
            raise ValueError("semantic_entropy edge model requires semantic probabilities.")
        eps = float(edge_cfg.get("entropy_eps", 1e-6))
        edges = -np.sum(
            semantic_probabilities * np.log(semantic_probabilities + eps), axis=0
        )
    elif method.startswith("torchvision_"):
        # Allow direct edge mapping via a torchvision segmentation model
        model_name = method.replace("torchvision_", "")
        semantic_model_cfg = {
            "name": model_name,
            "device": edge_cfg.get("device", "auto"),
            "output": edge_cfg.get("semantic_output", "max_prob"),
            "smooth_sigma": edge_cfg.get("semantic_smooth_sigma", 0.0),
            "entropy_eps": edge_cfg.get("entropy_eps", 1e-6),
            "target_class": edge_cfg.get("target_class"),
        }
        semantic_map, semantic_probabilities = _run_semantic_model_from_cfg(
            rgb_image, semantic_model_cfg
        )
        grad_y, grad_x = np.gradient(semantic_map)
        edges = np.sqrt(grad_x ** 2 + grad_y ** 2)
    elif method == "canny_rgb":
        sigma = float(edge_cfg.get("canny_sigma", 1.0))
        edges = canny(rgb2gray(rgb_image), sigma=sigma).astype(np.float64)
    else:
        raise ValueError(f"Unsupported edge_model.type value: {method}")

    if edge_cfg.get("normalize", True):
        edges = _normalize_map(edges)

    smooth_sigma = float(edge_cfg.get("smooth_sigma", 0.0))
    if smooth_sigma > 0.0:
        edges = gaussian(edges, sigma=smooth_sigma, mode="reflect")

    return np.clip(edges, 0.0, None), semantic_map, semantic_probabilities


def _blend_with_semantic(edges: np.ndarray, semantic_map: np.ndarray, semantic_cfg: Dict[str, object]) -> np.ndarray:
    """Blend semantic attention into the edge magnitude map."""

    mode = str(semantic_cfg.get("blend_mode", "multiply")).lower()
    weight = float(semantic_cfg.get("weight", 0.5))
    weight = max(0.0, weight)

    if mode == "multiply":
        factor = (1.0 - weight) + weight * semantic_map
        return edges * factor
    if mode == "replace":
        return (1.0 - weight) * edges + weight * semantic_map
    if mode == "max":
        multiplier = weight if weight > 0 else 1.0
        return np.maximum(edges, semantic_map * multiplier)
    # Default to additive blend
    return edges + weight * semantic_map


def compute_edge_guidance_weights(
    rgb_image: np.ndarray,
    config: Dict[str, object],
) -> EdgeGuidanceWeights:
    """Compute edge-aware sampling weights from an RGB image.

    Args:
        rgb_image: Float image in [0, 1] with shape (H, W, 3).
        config: Configuration dictionary with optional keys:
            - enabled (bool): Whether guidance is enabled (caller usually checks
              this before calling).
            - method (str): Edge detector ('sobel', 'scharr', 'prewitt', 'canny').
            - gaussian_sigma (float): Sigma for Gaussian smoothing before edges.
            - canny_sigma (float): Sigma for Canny if method == 'canny'.
            - exponent (float): Exponent applied to edge magnitude to sharpen contrast.
            - uniform_mix (float): Blend ratio with uniform distribution.
            - min_value (float): Floor added before normalization.
            - multi_scale (dict): Optional multi-scale configuration with keys such as
              'enabled', 'scales', 'weights', 'gaussian_sigmas', 'combine'.
            - semantic (dict): Optional semantic guidance settings ('enabled', 'path',
              'weight', 'blend_mode', etc.).

    Returns:
        EdgeGuidanceWeights containing per-domain probability vectors.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3)")

    method = str(config.get("method", "sobel")).lower()
    gaussian_sigma = float(config.get("gaussian_sigma", 0.8))
    exponent = float(config.get("exponent", 1.5))
    uniform_mix = float(config.get("uniform_mix", 0.15))
    min_value = float(config.get("min_value", 1e-4))
    canny_sigma = float(config.get("canny_sigma", 1.0))

    gray = rgb2gray(rgb_image)
    artifacts: Dict[str, np.ndarray] = {
        "gray": np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    }

    professional_cfg = config.get("professional_preprocessing") if isinstance(config.get("professional_preprocessing"), dict) else None
    edge_override: Optional[np.ndarray] = None
    semantic_override: Optional[np.ndarray] = None
    semantic_probabilities: Optional[np.ndarray] = None

    if isinstance(professional_cfg, dict) and professional_cfg.get("enabled", False):
        strict_errors = bool(professional_cfg.get("fail_on_missing", False))

        try:
            edge_override = _load_external_map(
                professional_cfg.get("edge_map_path"),
                gray.shape,
                normalize=bool(professional_cfg.get("edge_normalize", False)),
                clip_low=professional_cfg.get("edge_clip_low"),
                clip_high=professional_cfg.get("edge_clip_high"),
                fail_on_missing=strict_errors,
            )
            if edge_override is not None:
                artifacts["edge_magnitude_professional_file"] = np.clip(edge_override, 0.0, None)
        except Exception as exc:
            if strict_errors:
                raise
            warnings.warn(f"Failed to load professional edge map: {exc}")

        try:
            semantic_override = _load_external_map(
                professional_cfg.get("semantic_map_path"),
                gray.shape,
                normalize=bool(professional_cfg.get("semantic_normalize", True)),
                clip_low=professional_cfg.get("semantic_clip_low"),
                clip_high=professional_cfg.get("semantic_clip_high"),
                fail_on_missing=strict_errors,
            )
            if semantic_override is not None:
                artifacts["semantic_map_professional_file"] = np.clip(semantic_override, 0.0, None)
        except Exception as exc:
            if strict_errors:
                raise
            warnings.warn(f"Failed to load professional semantic map: {exc}")

        generated_semantic: Optional[np.ndarray] = None
        generated_edge: Optional[np.ndarray] = None

        semantic_model_cfg = professional_cfg.get("semantic_model")
        if isinstance(semantic_model_cfg, dict) and semantic_model_cfg.get("enabled", False):
            try:
                generated_semantic, semantic_probabilities = _run_semantic_model_from_cfg(rgb_image, semantic_model_cfg)
                artifacts["semantic_map_professional_model"] = np.clip(generated_semantic, 0.0, None)
            except Exception as exc:
                semantic_probabilities = None
                if strict_errors:
                    raise
                warnings.warn(f"Semantic model preprocessing failed: {exc}")

        if semantic_override is None and generated_semantic is not None:
            semantic_override = generated_semantic

        edge_model_cfg = professional_cfg.get("edge_model")
        if isinstance(edge_model_cfg, dict) and edge_model_cfg.get("enabled", False):
            try:
                generated_edge, generated_semantic_from_edge, generated_probs_from_edge = _generate_edge_map_from_cfg(
                    rgb_image,
                    edge_model_cfg,
                    semantic_override,
                    semantic_probabilities,
                )
                if generated_edge is not None:
                    artifacts["edge_magnitude_professional_model"] = np.clip(generated_edge, 0.0, None)
                if generated_semantic_from_edge is not None and semantic_override is None:
                    semantic_override = generated_semantic_from_edge
                if generated_probs_from_edge is not None and semantic_probabilities is None:
                    semantic_probabilities = generated_probs_from_edge
            except Exception as exc:
                if strict_errors:
                    raise
                warnings.warn(f"Edge model preprocessing failed: {exc}")

        if edge_override is None and generated_edge is not None:
            edge_override = generated_edge

    if edge_override is not None:
        edges = np.clip(edge_override, 0.0, None)
        artifacts["edge_magnitude_base"] = edges.copy()
    else:
        multi_cfg = config.get("multi_scale") if isinstance(config.get("multi_scale"), dict) else None
        if isinstance(multi_cfg, dict) and multi_cfg.get("enabled", False):
            base_edges = _compute_multi_scale_edges(gray, method, gaussian_sigma, canny_sigma, multi_cfg)
            edges = np.clip(base_edges, 0.0, None)
            artifacts["edge_magnitude_base"] = edges.copy()
        else:
            working_gray = gray
            if gaussian_sigma > 0.0:
                working_gray = gaussian(working_gray, sigma=gaussian_sigma, mode="reflect")
                artifacts["gray_smoothed"] = np.clip(working_gray, 0.0, None)
            base_edges = _compute_core_edges(working_gray, method, canny_sigma)
            edges = np.clip(base_edges, 0.0, None)
            artifacts["edge_magnitude_base"] = edges.copy()

    if exponent != 1.0:
        edges = np.power(edges, exponent)

    edges = np.clip(edges, 0.0, None)
    artifacts["edge_magnitude_post_exponent"] = edges.copy()

    semantic_cfg = config.get("semantic") if isinstance(config.get("semantic"), dict) else None
    if semantic_override is not None:
        base_cfg = semantic_cfg if isinstance(semantic_cfg, dict) else {}
        semantic_cfg = dict(base_cfg)
        semantic_cfg["array"] = semantic_override
        semantic_cfg.pop("path", None)
        semantic_cfg["enabled"] = True
        if isinstance(professional_cfg, dict) and "semantic_normalize" in professional_cfg:
            semantic_cfg["normalize"] = professional_cfg["semantic_normalize"]

    if isinstance(semantic_cfg, dict) and semantic_cfg.get("enabled", False):
        semantic_map = _load_semantic_map(semantic_cfg, gray.shape)
        if semantic_map is not None:
            artifacts["semantic_map"] = np.clip(semantic_map, 0.0, None)
            edges = _blend_with_semantic(edges, semantic_map, semantic_cfg)
            edges = np.clip(edges, 0.0, None)
            artifacts["edge_magnitude_post_semantic"] = edges.copy()
        elif semantic_cfg.get("fail_on_missing", False):
            raise FileNotFoundError("Semantic guidance enabled but no semantic map could be loaded.")

    edges = np.clip(edges, 0.0, None)

    pixel_weights = _normalize_weights(edges.reshape(-1), min_value, uniform_mix)
    channel_weights = np.repeat(pixel_weights / 3.0, 3)

    try:
        artifacts["normalized_weights"] = pixel_weights.reshape(gray.shape)
    except Exception:
        pass

    return EdgeGuidanceWeights(
        pixel_weights=pixel_weights,
        channel_weights=channel_weights,
        artifacts=artifacts,
    )


# =============================================================================
# DATA LOADER MODULE
# =============================================================================

def load_target_image_and_model(
    image_id: int, model_path: Path, data_root: Path, mode: str
) -> Tuple[np.ndarray, Optional[np.ndarray], int, nn.Module, List[str]]:
    """
    Load target image, true label and pretrained model.

    This function loads the specified index image from CIFAR-10 test set, converts
    it to appropriate format for model input, and loads pretrained ResNet18 model.
    Also processes image data according to perturbation mode.

    Args:
        image_id: CIFAR-10 test set image index, range [0, 9999]
        model_path: Path to pretrained model weights file
        data_root: Path to CIFAR-10 dataset root directory
        mode: Perturbation mode, determines whether to return additional V channel data
             - "v_channel": Return original V channel data
             - Other modes: V channel returns None

    Returns:
        original_rgb: Original RGB image array, shape [32, 32, 3], value range [0, 1]
        original_v: Original V channel array (v_channel mode only), shape [32, 32], value range [0, 1];
                   None for other modes
        true_label: True class index of image (0-9)
        model: Loaded ResNet18 model, moved to appropriate device and set to evaluation mode
        class_names: List of 10 CIFAR-10 class names

    Raises:
        RuntimeError: If unable to load model weights or dataset
    """
    # Define image preprocessing transform: convert to tensor
    transform = T.Compose([T.ToTensor()])

    # Load CIFAR-10 test set
    # train=False means load test set, download=True will auto-download if needed
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )

    # Get image and label at specified index
    img_tensor, true_label = test_set[image_id]

    # Determine compute device: prefer CUDA GPU, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)

    # Convert tensor to NumPy array, shape from [C, H, W] to [H, W, C], value range [0, 1]
    original_rgb_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    # Process V channel data according to perturbation mode
    original_v_np = None
    if mode == "v_channel":
        # Convert RGB to HSV color space
        original_hsv_np = rgb_to_hsv(original_rgb_np)
        # Extract V (brightness) channel
        original_v_np = original_hsv_np[:, :, 2]  # shape [32, 32]

    # Print loading information
    print(f"Loaded image id {image_id}, true label: {test_set.classes[true_label]} ({true_label})")

    # Load ResNet18 model
    # weights=None means don't use pretrained weights, we'll manually load custom weights
    model = resnet18(weights=None, num_classes=10)

    # Modify first convolution layer to adapt to 32x32 input (CIFAR-10 image size)
    # Original ResNet18 is for 224x224 images, here adjust to 3x3 conv, stride=1, padding=1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove max pooling layer because CIFAR-10 images are already small
    model.maxpool = nn.Identity()

    # Load custom trained weights
    # map_location=device ensures weights are loaded to correct device
    # weights_only=True improves security, only load weight parameters
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    # Move model to compute device and set to evaluation mode (disable dropout and batch norm updates)
    model.to(device).eval()

    # Disable gradient computation since this is inference phase
    model.requires_grad_(False)

    # Return all necessary data
    return original_rgb_np, original_v_np, true_label, model, test_set.classes


# =============================================================================
# OBJECTIVES MODULE
# =============================================================================

def evaluate_objectives_batch(
    population: List[Tuple[np.ndarray, np.ndarray]],
    original_rgb: np.ndarray,
    original_v: Optional[np.ndarray],
    true_label: int,
    model: nn.Module,
    targeted: bool,
    target_label: Optional[int],
    mode: str
) -> List[Tuple[bool, float, float, int]]:
    """
    Batch evaluate objective function values for individuals in population.

    This function performs batch inference on entire population, computing
    adversarial success, loss value, L2 norm and L0 norm for each individual.
    Supports both targeted and non-targeted attack modes.

    Args:
        population: Population list, each individual is (indices, perturbations) tuple
        original_rgb: Original RGB image, shape [32, 32, 3], value range [0, 1]
        original_v: Original V channel (v_channel mode only), shape [32, 32], value range [0, 1]
        true_label: True class index of original image
        model: Pretrained classification model
        targeted: Whether this is targeted attack
        target_label: Target class index (only used when targeted=True)
        mode: Perturbation mode ("rgb_sim", "channel", "v_channel")

    Returns:
        Objective value list, each element is (is_adversarial, loss, l2_norm, l0_norm) tuple
    """
    pop_size = len(population)
    device = next(model.parameters()).device
    batch_tensors = []

    # Build batch input tensors
    for indices, perturbations in population:
        if mode == "v_channel":
            # V channel perturbation: modify V channel of HSV
            noise_v = np.zeros_like(original_v, dtype=np.float32)
            noise_v.flat[indices] = perturbations
            perturbed_v = np.clip(original_v + noise_v, 0.0, 1.0)
            original_hsv = rgb_to_hsv(original_rgb)
            perturbed_hsv = np.stack([
                original_hsv[:, :, 0],  # H channel unchanged
                original_hsv[:, :, 1],  # S channel unchanged
                perturbed_v              # V channel with perturbation
            ], axis=-1)
            perturbed_rgb = hsv_to_rgb(perturbed_hsv)
            perturbed_rgb = np.clip(perturbed_rgb, 0.0, 1.0).astype(np.float32)
        else:
            # RGB or channel perturbation: directly apply perturbation in RGB space
            noise_rgb = np.zeros_like(original_rgb, dtype=np.float32)
            noise_rgb.flat[indices] = perturbations
            perturbed_rgb = np.clip(original_rgb + noise_rgb, 0.0, 1.0).astype(np.float32)

        # Apply real world robustness preprocessing (if enabled)
        if ENABLE_REAL_WORLD_ROBUSTNESS:
            perturbed_rgb = apply_real_world_preprocessing(perturbed_rgb)

        # Convert to PyTorch tensor format [C, H, W]
        tensor = torch.from_numpy(perturbed_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
        batch_tensors.append(tensor)

    # Batch inference
    batch_input = torch.cat(batch_tensors, dim=0)
    with torch.no_grad():
        outputs = model(batch_input)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

    objective_values = []
    for i in range(pop_size):
        pred_label = int(np.argmax(probs[i]))
        is_adversarial = False
        loss = 0.0

        if targeted:
            # Targeted attack: check if predicted as target class
            if target_label is None:
                raise ValueError("Target label must be provided for targeted attack.")
            is_adversarial = (pred_label == target_label)
            # Loss: negative log probability (minimize to maximize target class probability)
            log_probs = torch.log_softmax(outputs[i:i+1], dim=1).cpu().squeeze()
            loss = -float(log_probs[target_label].item())
        else:
            # Non-targeted attack: check if prediction is wrong
            is_adversarial = (pred_label != true_label)
            # Loss: difference between correct class probability and highest wrong class probability (margin loss)
            p_correct = float(probs[i, true_label])
            p_others = probs[i].copy()
            p_others[true_label] = -np.inf  # Exclude correct class
            p_max_other = float(np.max(p_others))
            loss = p_correct - p_max_other

        # Calculate L2 and L0 norms
        if mode == "v_channel":
            # V channel mode: perturbation only on V channel
            noise_full = np.zeros_like(original_v, dtype=np.float32)
            noise_full.flat[indices] = perturbations
            l2_norm = float(np.linalg.norm(noise_full))  # L2 norm
            l0_norm = int(np.count_nonzero(perturbations))  # Number of non-zero perturbations
        else:
            # RGB mode: perturbation in RGB space
            noise_full = np.zeros_like(original_rgb, dtype=np.float32)
            noise_full.flat[indices] = perturbations
            l2_norm = float(np.linalg.norm(noise_full))  # L2 norm
            if mode == "rgb_sim":
                # RGB synchronous mode: L0 is number of perturbed pixels
                pert_reshaped = perturbations.reshape(-1, 3)  # (k, 3)
                l0_norm = int(np.sum(np.any(pert_reshaped != 0, axis=1)))
            else:  # "channel"
                # Channel mode: L0 is number of perturbed channels
                l0_norm = int(np.count_nonzero(perturbations))

        objective_values.append((is_adversarial, loss, l2_norm, l0_norm))

    return objective_values


def apply_real_world_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Apply real world preprocessing to image: optional JPEG compression and Resize + Interpolation.

    Args:
        image: Input image, shape [H, W, C], value range [0, 1]

    Returns:
        Preprocessed image, shape [H, W, C], value range [0, 1]
    """
    # Convert to PIL Image
    pil_image = Image.fromarray((image * 255).astype(np.uint8))

    # JPEG compression (if quality < 100)
    if JPEG_QUALITY < 100:
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=JPEG_QUALITY)
        buffer.seek(0)
        pil_image = Image.open(buffer)

    # Resize + Interpolation (if enabled)
    if ENABLE_RESIZE_PREPROCESSING:
        original_size = pil_image.size  # (W, H)
        # Calculate reduced size
        new_size = (int(original_size[0] / RESIZE_SCALE), int(original_size[1] / RESIZE_SCALE))
        
        # Check if new size is valid
        if new_size[0] <= 0 or new_size[1] <= 0:
            print(f"Warning: Resize scale {RESIZE_SCALE} too large for {original_size} image, skipping resize preprocessing")
        else:
            # Downscale
            resized_image = pil_image.resize(new_size, Image.BILINEAR)
            # Upscale back to original size
            pil_image = resized_image.resize(original_size, Image.BILINEAR)

    # Convert back to numpy array
    processed_array = np.array(pil_image, dtype=np.float32) / 255.0
    return processed_array


def _objective_to_dict(obj: Tuple[bool, float, float, int]) -> Dict[str, Any]:
    l0_value = obj[3]
    if isinstance(l0_value, (int, np.integer)):
        l0_norm = int(l0_value)
    elif isinstance(l0_value, float) and math.isfinite(l0_value):
        l0_norm = int(round(l0_value))
    else:
        l0_norm = float(l0_value)

    return {
        "is_adversarial": bool(obj[0]),
        "loss": float(obj[1]),
        "l2_norm": float(obj[2]),
        "l0_norm": l0_norm
    }


def _match_condition(condition: Dict[str, Any], self_metrics: Dict[str, Any], other_metrics: Dict[str, Any]) -> bool:
    if not condition:
        return True
    for key, expected in condition.items():
        if key.startswith("other_"):
            metric = key[len("other_"):]
            if other_metrics.get(metric) != expected:
                return False
        else:
            if self_metrics.get(key) != expected:
                return False
    return True


def _normalize_scalar(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _compare_metric(rule: Dict[str, Any], metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> int:
    name = rule.get("name")
    if not name:
        return 0
    if name not in metrics_a or name not in metrics_b:
        return 0

    goal = str(rule.get("goal", "min")).lower()
    tolerance = float(rule.get("tolerance", 0.0) or 0.0)
    value_a = _normalize_scalar(metrics_a[name])
    value_b = _normalize_scalar(metrics_b[name])
    diff = value_a - value_b

    if goal == "min":
        if diff < -tolerance:
            return 1
        if diff > tolerance:
            return -1
    elif goal == "max":
        if diff > tolerance:
            return 1
        if diff < -tolerance:
            return -1

    return 0


def _legacy_compare(obj1: Tuple[bool, float, float, int], obj2: Tuple[bool, float, float, int]) -> int:
    is_adv1, loss1, l2_1, l0_1 = obj1
    is_adv2, loss2, l2_2, l0_2 = obj2

    if is_adv1 and not is_adv2:
        return 1
    if not is_adv1 and is_adv2:
        return -1
    if is_adv1 and is_adv2:
        if l2_1 < l2_2:
            return 1
        if l2_1 > l2_2:
            return -1
        if l0_1 < l0_2:
            return 1
        if l0_1 > l0_2:
            return -1
    if not is_adv1 and not is_adv2:
        if loss1 < loss2:
            return 1
        if loss1 > loss2:
            return -1
    return 0


def _compare_with_config(obj1: Tuple[bool, float, float, int], obj2: Tuple[bool, float, float, int], config: Dict[str, Any]) -> int:
    metrics_a = _objective_to_dict(obj1)
    metrics_b = _objective_to_dict(obj2)

    for rule in config.get("rules", []):
        condition = rule.get("when", {})
        if _match_condition(condition, metrics_a, metrics_b):
            prefer = rule.get("prefer")
            if prefer == "self":
                return 1
            if prefer == "other":
                return -1

            for metric_rule in rule.get("metrics", []):
                verdict = _compare_metric(metric_rule, metrics_a, metrics_b)
                if verdict != 0:
                    return verdict
            # Condition matched but no clear preference, continue to next rule

    # If rules are undecided, try tie breakers or fallback strategy
    for metric_rule in config.get("tie_breakers", []):
        verdict = _compare_metric(metric_rule, metrics_a, metrics_b)
        if verdict != 0:
            return verdict

    return _legacy_compare(obj1, obj2)


def dominates(obj1: Tuple[bool, float, float, int], obj2: Tuple[bool, float, float, int]) -> bool:
    """
    Determine if obj1 dominates obj2 (dominance relation definition from paper section 3.1).

    Dominance relation definition:
    - If obj1 is adversarial while obj2 is not, then obj1 dominates obj2
    - If both are adversarial, compare L2 norm, smaller dominates
    - If both are not adversarial, compare loss value, smaller dominates

    Args:
        obj1: First objective value tuple (is_adversarial, loss, l2_norm, l0_norm)
        obj2: Second objective value tuple (is_adversarial, loss, l2_norm, l0_norm)

    Returns:
        True if obj1 dominates obj2, otherwise False
    """
    verdict = _compare_with_config(obj1, obj2, DOMINANCE_CONFIG)
    if verdict > 0:
        return True
    if verdict < 0:
        return False
    # In tie case, check by reverse comparison to ensure consistency
    reverse_verdict = _compare_with_config(obj2, obj1, DOMINANCE_CONFIG)
    return reverse_verdict < 0


def non_dominated_sort(objective_values: List[Tuple[bool, float, float, int]]) -> List[List[int]]:
    """
    Perform non-dominated sorting, return index lists of each front.

    Non-dominated sorting divides population into multiple fronts:
    - Front 0: Non-dominated solutions (no other solutions dominate them)
    - Front 1: Non-dominated among solutions dominated by front 0, and so on

    Args:
        objective_values: Objective value list of all individuals

    Returns:
        Front list, each front contains indices of individuals in that front
    """
    pop_size = len(objective_values)
    fronts: List[List[int]] = [[]]  # Initialize front 0

    # domination_info[i] = {"n": dominated by how many individuals, "S": list of dominated individual indices}
    domination_info = [{"n": 0, "S": []} for _ in range(pop_size)]

    # Calculate dominance relations
    for p in range(pop_size):
        for q in range(p + 1, pop_size):
            if dominates(objective_values[p], objective_values[q]):
                domination_info[p]["S"].append(q)
                domination_info[q]["n"] += 1
            elif dominates(objective_values[q], objective_values[p]):
                domination_info[q]["S"].append(p)
                domination_info[p]["n"] += 1

        # If not dominated by any individual, add to front 0
        if domination_info[p]["n"] == 0:
            fronts[0].append(p)

    # Build subsequent fronts
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in domination_info[p]["S"]:
                domination_info[q]["n"] -= 1
                if domination_info[q]["n"] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


# =============================================================================
# DYNAMIC PARAMETERS MODULE
# =============================================================================

def _clamp(value: float, min_value: Optional[float], max_value: Optional[float]) -> float:
    if min_value is not None and value < min_value:
        value = min_value
    if max_value is not None and value > max_value:
        value = max_value
    return value


@dataclass
class DynamicParameterScheduler:
    """Scheduler that produces per-generation parameter values."""

    config: Dict[str, Any]
    total_generations: int
    is_targeted_attack: bool

    def __post_init__(self) -> None:
        self.total_generations = max(int(self.total_generations), 1)
        if not isinstance(self.config, dict):
            self.config = {}

    def get_params(self, generation: int) -> Dict[str, float]:
        """Return all dynamic parameter values for the given generation index."""
        progress = self._compute_progress(generation)
        result: Dict[str, float] = {}
        for name, cfg in self.config.items():
            result[name] = self._compute_value(cfg or {}, progress)
        return result

    def get_value(self, name: str, generation: int, default: Optional[float] = None) -> Optional[float]:
        """Return a single dynamic parameter value, or the provided default if missing."""
        cfg = self.config.get(name)
        if cfg is None:
            return default
        return self._compute_value(cfg, self._compute_progress(generation))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_progress(self, generation: int) -> float:
        if self.total_generations <= 1:
            return 0.0
        ratio = float(generation) / float(self.total_generations - 1)
        return max(0.0, min(1.0, ratio))

    def _compute_value(self, cfg: Dict[str, Any], progress: float) -> float:
        schedule = str(cfg.get("schedule", "constant")).lower()
        if schedule == "linear":
            start = float(self._resolve_contextual_value(cfg, "start", cfg.get("value", 0.0)))
            end = float(self._resolve_contextual_value(cfg, "end", start))
            value = start + (end - start) * progress
        elif schedule == "exponential":
            start = float(self._resolve_contextual_value(cfg, "start", cfg.get("value", 0.0)))
            end = float(self._resolve_contextual_value(cfg, "end", start))
            if start <= 0 or end <= 0:
                value = start + (end - start) * progress
            else:
                value = start * ((end / start) ** progress)
        elif schedule == "piecewise":
            value = self._compute_piecewise(cfg, progress)
        else:
            value = float(self._resolve_contextual_value(cfg, "value", cfg.get("default", 0.0)))

        min_val = self._resolve_contextual_value(cfg, "min")
        max_val = self._resolve_contextual_value(cfg, "max")
        value = _clamp(value, min_val, max_val)

        precision = self._resolve_contextual_value(cfg, "precision")
        if precision is not None:
            try:
                digits = int(precision)
                value = round(value, digits)
            except (TypeError, ValueError):
                pass

        return value

    def _compute_piecewise(self, cfg: Dict[str, Any], progress: float) -> float:
        pieces = self._resolve_contextual_value(cfg, "pieces")
        if pieces is None:
            pieces = cfg.get("pieces")
        if not pieces:
            fallback = self._resolve_contextual_value(cfg, "value")
            return float(fallback) if fallback is not None else 0.0

        normalized: list[tuple[float, float]] = []
        for piece in pieces:
            if not isinstance(piece, dict):
                continue
            threshold = piece.get("progress")
            if threshold is None:
                threshold = piece.get("until")
            try:
                threshold_f = float(threshold)
            except (TypeError, ValueError):
                threshold_f = 1.0

            piece_value = self._resolve_contextual_value(piece, "value")
            if piece_value is None:
                piece_value = piece.get("value")
            if piece_value is None:
                continue
            try:
                normalized.append((threshold_f, float(piece_value)))
            except (TypeError, ValueError):
                continue

        if not normalized:
            fallback = self._resolve_contextual_value(cfg, "value", 0.0)
            return float(fallback)

        normalized.sort(key=lambda item: item[0])
        for threshold, value in normalized:
            if progress <= threshold:
                return value
        return normalized[-1][1]

    def _resolve_contextual_value(self, cfg: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
        if self.is_targeted_attack:
            targeted_block = cfg.get("targeted", {})
            if isinstance(targeted_block, dict) and key in targeted_block:
                return targeted_block[key]
            targeted_key = f"targeted_{key}"
            if targeted_key in cfg:
                return cfg[targeted_key]
        else:
            non_targeted_block = cfg.get("non_targeted", {})
            if isinstance(non_targeted_block, dict) and key in non_targeted_block:
                return non_targeted_block[key]
            non_targeted_key = f"non_targeted_{key}"
            if non_targeted_key in cfg:
                return cfg[non_targeted_key]

        return cfg.get(key, default)


# =============================================================================
# EVOLUTIONARY OPERATORS MODULE
# =============================================================================

def _sample_full_domain(
    mode: str,
    domain_size: int,
    sample_size: int,
    edge_guidance: Optional[EdgeGuidanceWeights],
) -> np.ndarray:
    """Sample indices from the full domain, optionally using edge guidance."""
    sample_size = min(sample_size, domain_size)
    if edge_guidance is not None and sample_size > 0:
        candidates = np.arange(domain_size, dtype=int)
        return edge_guidance.sample_without_replacement(mode, candidates, sample_size)
    return np.random.choice(domain_size, size=sample_size, replace=False)


def _sample_from_candidates(
    mode: str,
    candidates: np.ndarray,
    sample_size: int,
    edge_guidance: Optional[EdgeGuidanceWeights],
) -> np.ndarray:
    """Sample indices from a candidate set with optional guidance."""
    if candidates.size == 0 or sample_size <= 0:
        return np.empty(0, dtype=int)

    sample_size = min(sample_size, candidates.size)
    if edge_guidance is not None:
        return edge_guidance.sample_without_replacement(mode, candidates, sample_size)
    return np.random.choice(candidates, size=sample_size, replace=False)


def generate_perturbation_values(size: int, zero_sample_prob: float = ZERO_SAMPLE_PROB) -> np.ndarray:
    """
    Generate perturbation values according to configuration.

    If continuous perturbation is enabled, randomly select n decimal places from [lower_bound, upper_bound].
    Otherwise, select discrete values from {-1, 0, 1}.

    Args:
        size: Number of perturbation values to generate

    Returns:
        Perturbation value array
    """
    if ENABLE_CONTINUOUS_PERTURBATION:
        # Continuous perturbation: randomly select from specified range
        values = np.random.uniform(CONTINUOUS_LOWER_BOUND, CONTINUOUS_UPPER_BOUND, size)
        # Round to specified decimal places
        values = np.round(values, CONTINUOUS_DECIMAL_PLACES)
        return values
    else:
        # Discrete perturbation: select from {-1, 0, 1}
        stay_prob = np.clip(zero_sample_prob, 0.0, 1.0)
        side_prob = (1 - stay_prob) / 2
        return np.random.choice([-1, 0, 1], size=size,
                               p=[side_prob, stay_prob, side_prob])


def initialize_population(
    pop_size: int,
    k: int,
    mode: str,
    zero_sample_prob: float = ZERO_SAMPLE_PROB,
    edge_guidance: Optional[EdgeGuidanceWeights] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Initialize population according to perturbation mode.

    Generate initial population for evolutionary algorithm, each individual contains
    perturbation position indices and perturbation values. Supports three perturbation
    modes, each with different index and value initialization strategies.

    Args:
        pop_size: Population size (number of individuals)
        k: Perturbation budget (maximum number allowed to modify)
        mode: Perturbation mode
             - "v_channel": Perturb V channel of HSV
             - "rgb_sim": RGB synchronous perturbation (pixel-level)
             - "channel": Independent channel perturbation

    Returns:
        Initialized population list, each individual is (indices, perturbations) tuple
    """
    population = []
    total_pixels = 32 * 32  # CIFAR-10 image size
    total_channels = total_pixels * 3  # RGB three channels

    for _ in range(pop_size):
        if mode == "v_channel":
            # V channel mode: select k pixel positions, perturbation values are -1,0,1
            indices = _sample_full_domain("v_channel", total_pixels, k, edge_guidance)
            values = generate_perturbation_values(k, zero_sample_prob)
            population.append((indices, values))

        elif mode == "rgb_sim":
            # RGB synchronous mode: select k pixels, RGB three channels of each pixel are perturbed simultaneously
            pixel_indices = _sample_full_domain("rgb_sim", total_pixels, k, edge_guidance)
            # Generate 3 perturbation values for each pixel (RGB)
            values = generate_perturbation_values(k * 3, zero_sample_prob)
            # Expand pixel indices to RGB indices: pixel_idx -> [pixel_idx*3, pixel_idx*3+1, pixel_idx*3+2]
            rgb_indices = np.repeat(pixel_indices * 3, 3) + np.tile([0, 1, 2], k)
            population.append((rgb_indices, values))

        elif mode == "channel":
            # Channel mode: select k independent channel positions for perturbation
            channel_indices = _sample_full_domain("channel", total_channels, k, edge_guidance)
            values = generate_perturbation_values(k, zero_sample_prob)
            population.append((channel_indices, values))
        else:
            raise ValueError(f"Unknown PERTURBATION_MODE: {mode}")

    return population


def crossover_v_channel(ind1, ind2, k, crossover_prob: float = CROSSOVER_PROB):
    """
    V channel specific crossover operator.

    Implement crossover operation defined in paper: exchange partial perturbation
    positions from two parent individuals. Only exchange perturbation values 
    corresponding to positions that ind1 has but ind2 doesn't.

    Args:
        ind1: Parent individual 1 (indices1, perturbations1)
        ind2: Parent individual 2 (indices2, perturbations2)
        k: Perturbation budget

    Returns:
        Two offspring individuals after crossover (child1, child2)
    """
    ma, da = ind1  # indices and perturbations of parent1
    mb, db = ind2

    # Safety check: ensure individual structure is correct
    assert len(ma) == len(da), f"Individual p1 corrupted: len(ma)={len(ma)} != len(da)={len(da)}"
    assert len(mb) == len(db), f"Individual p2 corrupted: len(mb)={len(mb)} != len(db)={len(db)}"

    # Find pixel positions that ind2 has but ind1 doesn't
    set_ma = set(ma)
    set_mb = set(mb)
    u = list(set_mb - set_ma)  # Positions unique to ind2

    if len(u) == 0:
        return ind1, ind2  # No exchangeable positions, return original individuals

    # Calculate maximum exchange count: crossover_prob * k
    max_exchange = int(crossover_prob * k)
    exchange_count = min(max_exchange, len(u))

    # Randomly select positions to exchange
    a_indices = np.random.choice(len(ma), size=exchange_count, replace=False)
    b_indices = np.random.choice(len(u), size=exchange_count, replace=False)

    # Build offspring 1: remove selected positions, add positions from ind2
    new_ma = np.delete(ma, a_indices)
    new_b = np.array(u)[b_indices]
    new_ma = np.concatenate([new_ma, new_b])

    # Create mb to db index mapping to get perturbation values for corresponding positions
    mb_to_idx = {pixel: i for i, pixel in enumerate(mb)}
    indices_in_mb = [mb_to_idx[pixel] for pixel in new_b]  # Integer index list
    new_db_from_p2 = db[indices_in_mb]

    new_da = np.delete(da, a_indices)
    new_da = np.concatenate([new_da, new_db_from_p2])

    # Similarly build offspring 2
    new_mb = np.delete(mb, b_indices)
    new_a = ma[a_indices]
    new_mb = np.concatenate([new_mb, new_a])

    ma_to_idx = {pixel: i for i, pixel in enumerate(ma)}
    indices_in_ma = [ma_to_idx[pixel] for pixel in new_a]
    new_da_from_p1 = da[indices_in_ma]

    new_db = np.delete(db, b_indices)
    new_db = np.concatenate([new_db, new_da_from_p1])

    return (new_ma, new_da), (new_mb, new_db)


def crossover(ind1, ind2, k, mode, crossover_prob: float = CROSSOVER_PROB):
    """
    Select appropriate crossover operator according to perturbation mode.

    Args:
        ind1: Parent individual 1
        ind2: Parent individual 2
        k: Perturbation budget
        mode: Perturbation mode

    Returns:
        Two offspring individuals after crossover
    """
    if mode == "v_channel":
        return crossover_v_channel(ind1, ind2, k, crossover_prob)
    elif mode == "rgb_sim":
        # RGB synchronous mode crossover operator
        ma, da = ind1
        mb, db = ind2

        # Convert RGB indices back to pixel indices
        pixel_ma = ma[::3] // 3  # Every 3 consecutive indices correspond to one pixel
        pixel_mb = mb[::3] // 3

        set_ma = set(pixel_ma)
        set_mb = set(pixel_mb)
        u = list(set_mb - set_ma)  # Pixels unique to ind2

        if len(u) == 0:
            return ind1, ind2

        max_exchange = int(crossover_prob * k)
        exchange_count = min(max_exchange, len(u))

        a_indices = np.random.choice(len(pixel_ma), size=exchange_count, replace=False)
        b_indices = np.random.choice(len(u), size=exchange_count, replace=False)

        # Build new pixel indices
        new_pixel_ma = np.delete(pixel_ma, a_indices)
        new_b = np.array(u)[b_indices]
        new_pixel_ma = np.concatenate([new_pixel_ma, new_b])

        # Convert back to RGB indices
        new_ma_rgb = np.repeat(new_pixel_ma * 3, 3) + np.tile([0, 1, 2], len(new_pixel_ma))

        # Handle perturbation values
        da_pixels = da.reshape(-1, 3)  # (k, 3)
        new_da_pixels = np.delete(da_pixels, a_indices, axis=0)

        mb_to_idx = {p: i for i, p in enumerate(pixel_mb)}
        indices_in_mb = [mb_to_idx[p] for p in new_b]
        b_da_from_p2 = db.reshape(-1, 3)[indices_in_mb]

        new_da = np.concatenate([new_da_pixels, b_da_from_p2]).flatten()

        # Similarly handle offspring 2
        new_pixel_mb = np.delete(pixel_mb, b_indices)
        new_a = pixel_ma[a_indices]
        new_pixel_mb = np.concatenate([new_pixel_mb, new_a])
        new_mb_rgb = np.repeat(new_pixel_mb * 3, 3) + np.tile([0, 1, 2], len(new_pixel_mb))

        db_pixels = db.reshape(-1, 3)
        new_db_pixels = np.delete(db_pixels, b_indices, axis=0)

        indices_in_ma = [np.where(pixel_ma == p)[0][0] for p in new_a]
        a_db_from_p1 = da.reshape(-1, 3)[indices_in_ma]

        new_db = np.concatenate([new_db_pixels, a_db_from_p1]).flatten()

        return (new_ma_rgb, new_da), (new_mb_rgb, new_db)

    else:  # "channel" mode
        ma, da = ind1
        mb, db = ind2

        set_ma = set(ma)
        set_mb = set(mb)
        u = list(set_mb - set_ma)

        if len(u) == 0:
            return ind1, ind2

        max_exchange = int(crossover_prob * k)
        exchange_count = min(max_exchange, len(u))

        a_indices = np.random.choice(len(ma), size=exchange_count, replace=False)
        b_indices = np.random.choice(len(u), size=exchange_count, replace=False)

        new_ma = np.delete(ma, a_indices)
        new_b = np.array(u)[b_indices]
        new_ma = np.concatenate([new_ma, new_b])

        new_da = np.delete(da, a_indices)

        mb_to_idx = {ch: i for i, ch in enumerate(mb)}
        indices_in_mb = [mb_to_idx[ch] for ch in new_b]
        b_da_from_p2 = db[indices_in_mb]

        new_da = np.concatenate([new_da, b_da_from_p2])

        new_mb = np.delete(mb, b_indices)
        new_a = ma[a_indices]
        new_mb = np.concatenate([new_mb, new_a])

        new_db = np.delete(db, b_indices)

        indices_in_ma = [np.where(ma == ch)[0][0] for ch in new_a]
        a_db_from_p1 = da[indices_in_ma]

        new_db = np.concatenate([new_db, a_db_from_p1])

        return (new_ma, new_da), (new_mb, new_db)


def mutate_v_channel(
    individual,
    k,
    pm,
    zero_sample_prob: float = ZERO_SAMPLE_PROB,
    edge_guidance: Optional[EdgeGuidanceWeights] = None,
):
    """
    V channel specific mutation operator.

    Change individual's perturbation pattern by adding new positions and removing existing positions.

    Args:
        individual: Individual to mutate (indices, perturbations)
        k: Perturbation budget
        pm: Mutation probability

    Returns:
        Mutated individual
    """
    ma, da = individual
    assert len(ma) == len(da), f"Mutate input corrupted: len(ma)={len(ma)} != len(da)={len(da)}"

    total_pixels = 32 * 32

    # If random number is greater than mutation probability, return original individual directly
    if np.random.rand() >= pm:
        return individual

    # Find unperturbed pixel positions
    set_ma = set(ma)
    t = list(set(range(total_pixels)) - set_ma)
    available_pixels = np.array(t, dtype=int)
    if available_pixels.size == 0:
        return individual

    # Calculate number to mutate: mutation_prob * k, but at least 1, at most available positions
    remove_count = max(1, min(int(pm * k), len(ma), available_pixels.size))
    add_count = remove_count

    # Randomly select positions to remove and add
    a_indices = np.random.choice(len(ma), size=remove_count, replace=False)
    new_pixels = _sample_from_candidates("v_channel", available_pixels, add_count, edge_guidance)

    # Build new indices and perturbations
    new_ma = np.delete(ma, a_indices)
    new_ma = np.concatenate([new_ma, new_pixels])

    new_da = np.delete(da, a_indices)
    new_da_b = generate_perturbation_values(add_count, zero_sample_prob)
    new_da = np.concatenate([new_da, new_da_b])

    return (new_ma, new_da)


def mutate(
    individual,
    k,
    pm,
    mode,
    zero_sample_prob: float = ZERO_SAMPLE_PROB,
    edge_guidance: Optional[EdgeGuidanceWeights] = None,
):
    """
    Select appropriate mutation operator according to perturbation mode.

    Args:
        individual: Individual to mutate
        k: Perturbation budget
        pm: Mutation probability
        mode: Perturbation mode

    Returns:
        Mutated individual
    """
    if mode == "v_channel":
        return mutate_v_channel(individual, k, pm, zero_sample_prob, edge_guidance)
    elif mode == "rgb_sim":
        ma, da = individual
        pixel_ma = ma[::3] // 3  # Pixel indices
        total_pixels = 32 * 32
        set_ma = set(pixel_ma)
        t = list(set(range(total_pixels)) - set_ma)
        available_pixels = np.array(t, dtype=int)
        if available_pixels.size == 0:
            return individual

        set_size = max(1, min(int(pm * k), len(pixel_ma), available_pixels.size))

        a_indices = np.random.choice(len(pixel_ma), size=set_size, replace=False)
        new_pixels = _sample_from_candidates("rgb_sim", available_pixels, set_size, edge_guidance)

        new_pixel_ma = np.delete(pixel_ma, a_indices)
        new_pixel_ma = np.concatenate([new_pixel_ma, new_pixels])

        new_ma_rgb = np.repeat(new_pixel_ma * 3, 3) + np.tile([0, 1, 2], len(new_pixel_ma))

        da_pixels = da.reshape(-1, 3)
        keep_mask = np.ones(len(pixel_ma), dtype=bool)
        keep_mask[a_indices] = False
        new_da_kept = da_pixels[keep_mask].flatten()

        new_da_b = generate_perturbation_values(set_size * 3, zero_sample_prob)
        new_da = np.concatenate([new_da_kept, new_da_b])

        return (new_ma_rgb, new_da)

    else:  # "channel"
        ma, da = individual
        total_channels = 32 * 32 * 3
        set_ma = set(ma)
        t = list(set(range(total_channels)) - set_ma)
        available_channels = np.array(t, dtype=int)
        if available_channels.size == 0:
            return individual

        set_size = max(1, min(int(pm * k), len(ma), available_channels.size))

        a_indices = np.random.choice(len(ma), size=set_size, replace=False)
        new_channels = _sample_from_candidates("channel", available_channels, set_size, edge_guidance)

        new_ma = np.delete(ma, a_indices)
        new_ma = np.concatenate([new_ma, new_channels])

        new_da = np.delete(da, a_indices)
        new_da_b = generate_perturbation_values(set_size, zero_sample_prob)
        new_da = np.concatenate([new_da, new_da_b])

        return (new_ma, new_da)


def selection(combined_population, objective_values, pop_size):
    """
    Selection operation based on non-dominated sorting.

    Select next generation population from combined parent and offspring population.

    Args:
        combined_population: Parent + offspring population
        objective_values: Corresponding objective values
        pop_size: Selected population size

    Returns:
        Selected population
    """
    fronts = non_dominated_sort(objective_values)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend([combined_population[i] for i in front])
        else:
            remaining = pop_size - len(selected)
            chosen = np.random.choice(front, remaining, replace=False)
            selected.extend([combined_population[i] for i in chosen])
            break

    return selected


# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

# Global variables for storing visualization window
fig: Optional[plt.Figure] = None
axs: Optional[np.ndarray] = None


def init_visualization(original_rgb: np.ndarray) -> None:
    """
    Initialize visualization window.

    Create a window with three subplots:
    - Left: Original image
    - Middle: Best noise heatmap
    - Right: Perturbed image

    Args:
        original_rgb: Original RGB image, shape [32, 32, 3], value range [0, 1]
    """
    global fig, axs
    plt.ion()  # Enable interactive mode
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display original image
    axs[0].imshow(original_rgb)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Reserve middle plot for noise display
    axs[1].set_title("Best Noise")
    axs[1].axis("off")

    # Reserve right plot for perturbed image display
    axs[2].set_title("Perturbed Image")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def update_visualization(
    original_rgb: np.ndarray,
    best_solution: Optional[Tuple[np.ndarray, np.ndarray]],
    best_obj: Tuple[bool, float, float, int],
    generation: int,
    mode: str,
    original_v: Optional[np.ndarray] = None
) -> None:
    """
    Update visualization window content.

    Update noise and perturbed image display according to current best solution.

    Args:
        original_rgb: Original RGB image
        best_solution: Current best solution (indices, perturbations), skip update if None
        best_obj: Best solution objective values (is_adversarial, loss, l2_norm, l0_norm)
        generation: Current generation
        mode: Perturbation mode
        original_v: Original V channel (v_channel mode only)
    """
    if not fig or not best_solution:
        return

    is_adv, loss, l2, l0 = best_obj
    indices, perturbations = best_solution

    # Reconstruct noise and perturbed image according to mode
    if mode == "v_channel":
        # V channel mode: perturbation only on brightness channel
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
        perturbed_rgb = np.clip(perturbed_rgb, 0.0, 1.0)

        # Noise display: normalize to [0,1] range
        noise_display = noise_full.copy()
        if noise_display.max() > noise_display.min():
            noise_display = (noise_display - noise_display.min()) / (noise_display.max() - noise_display.min())
    else:
        # RGB mode: perturbation in RGB space
        noise_full = np.zeros_like(original_rgb, dtype=np.float32)
        noise_full.flat[indices] = perturbations
        perturbed_rgb = np.clip(original_rgb + noise_full, 0.0, 1.0)

        # Noise display: use L2 norm as intensity
        noise_display = np.linalg.norm(noise_full, axis=2)
        if noise_display.max() > noise_display.min():
            noise_display = (noise_display - noise_display.min()) / (noise_display.max() - noise_display.min())

    # Update noise heatmap
    axs[1].clear()
    axs[1].imshow(noise_display, cmap="hot", interpolation="nearest")
    axs[1].set_title(f"Best Noise (Gen {generation})\nL2: {l2:.2f}, L0: {int(l0)}")
    axs[1].axis("off")

    # Update perturbed image
    axs[2].clear()
    axs[2].imshow(perturbed_rgb)
    status = "SUCCESS" if is_adv else "FAIL"
    axs[2].set_title(f"Perturbed (Gen {generation})\nStatus: {status}, Loss: {loss:.3f}")
    axs[2].axis("off")

    plt.draw()
    plt.pause(0.01)  # Brief pause to update display


def save_noise_heatmap(noise_channel: np.ndarray, channel_name: str, save_path) -> None:
    """
    Save noise heatmap to file.

    Args:
        noise_channel: Noise data array
        channel_name: Channel name (for title)
        save_path: Save path
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(noise_channel, cmap="hot", interpolation="nearest")
    plt.colorbar(im, shrink=0.8, label="Noise Magnitude")
    plt.title(f"Noise Heatmap - {channel_name} Channel")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_delta_L_heatmap(delta_L: np.ndarray, save_path) -> None:
    """
    Save ΔL* difference heatmap (CIELAB color space).

    Args:
        delta_L: ΔL* difference array, shape [32, 32]
        save_path: Save path
    """
    fig_deltaL, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(delta_L, cmap="RdBu_r", vmin=-5, vmax=5, interpolation="nearest")
    plt.colorbar(im, shrink=0.8, label="ΔL* (Lab)")
    ax.set_title("ΔL* Difference (Perturbed - Original)")
    ax.axis("off")
    plt.tight_layout()
    fig_deltaL.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig_deltaL)


def save_adversarial_result(
    original_rgb: np.ndarray,
    noise_display: np.ndarray,
    perturbed_rgb: np.ndarray,
    true_class_name: str,
    predicted_class_name: str,
    l0_norm: int,
    mode: str,
    save_path
) -> None:
    """
    Save final adversarial attack result plot.

    Create triptych containing original image, noise and perturbed image.

    Args:
        original_rgb: Original image
        noise_display: Noise display data (normalized)
        perturbed_rgb: Perturbed image
        true_class_name: Original class name
        predicted_class_name: Predicted class name
        l0_norm: L0 norm
        mode: Perturbation mode
        save_path: Save path
    """
    fig_final, axs_final = plt.subplots(1, 3, figsize=(15, 5))

    axs_final[0].imshow(original_rgb)
    axs_final[0].set_title(f"Original ({true_class_name})")
    axs_final[0].axis("off")

    axs_final[1].imshow(noise_display, cmap="hot", interpolation="nearest")
    axs_final[1].set_title(f"Best Noise ({mode.upper()})\nL0: {l0_norm}")
    axs_final[1].axis("off")

    axs_final[2].imshow(perturbed_rgb)
    axs_final[2].set_title(f"Perturbed ({predicted_class_name})")
    axs_final[2].axis("off")

    plt.tight_layout()
    fig_final.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig_final)


def save_convergence_plot(history_l2, history_l0, save_path) -> None:
    """
    Save convergence curve plot.

    Display L2 and L0 norm changes over generations.

    Args:
        history_l2: L2 norm history
        history_l0: L0 norm history
        save_path: Save path
    """
    fig_conv, ax1 = plt.subplots(figsize=(10, 5))

    # Filter successful generations (generations with adversarial samples)
    gens = np.arange(len(history_l2))
    successful_gens = [g for g, l2 in zip(gens, history_l2) if l2 is not None]
    successful_l2s = [l2 for l2 in history_l2 if l2 is not None]
    successful_l0s = [l0 for l0 in history_l0 if l0 is not None]

    if successful_l2s:
        color1 = "tab:red"
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("L2 Norm", color=color1)
        ax1.plot(successful_gens, successful_l2s, color=color1, label="L2 Norm")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(bottom=0)

        ax2 = ax1.twinx()
        color2 = "tab:blue"
        ax2.set_ylabel("L0 Norm", color=color2)
        ax2.plot(successful_gens, successful_l0s, color=color2, label="L0 Norm")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(bottom=0)

        fig_conv.tight_layout()
        plt.title("Convergence of L2 and L0 Norms (Successful Attacks)")
        plt.savefig(save_path)
    plt.close(fig_conv)


def save_edge_guidance_artifacts(
    artifacts: Dict[str, np.ndarray],
    save_dir,
    prefix: str = "semantic_guidance",
) -> None:
    """Save diagnostic visuals for edge/semantic guidance artifacts as a single overview image."""

    def _normalize_for_display(array: np.ndarray) -> np.ndarray:
        data = np.asarray(array, dtype=np.float64)
        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data_min = float(np.min(data)) if data.size else 0.0
        data_max = float(np.max(data)) if data.size else 0.0
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)
        return data

    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    mapping = [
        ("gray", "Grayscale Reference", "gray"),
        ("edge_magnitude_base", "Base Edge Magnitude", "magma"),
        ("edge_magnitude_post_exponent", "Edges After Exponent", "inferno"),
        ("semantic_map", "Semantic Map", "viridis"),
        ("edge_magnitude_post_semantic", "Edges After Semantic Blend", "magma"),
        ("normalized_weights", "Normalized Sampling Weights", "plasma"),
    ]

    entries = []
    for key, title, cmap in mapping:
        if key in artifacts and isinstance(artifacts[key], np.ndarray) and artifacts[key].size > 0:
            entries.append((artifacts[key], title, cmap))

    if not entries:
        return

    num_entries = len(entries)
    cols = min(3, num_entries)
    cols = max(cols, 1)
    rows = (num_entries + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.atleast_2d(axes)
    elif cols == 1:
        axes = np.atleast_2d(axes).T

    for idx, (array, title, cmap) in enumerate(entries):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        normalized = _normalize_for_display(array)
        im = ax.imshow(normalized, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Hide any unused axes
    for idx in range(num_entries, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    fig.tight_layout()
    output_path = target_dir / f"{prefix}_overview.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN EXECUTION MODULE
# =============================================================================

def main() -> None:
    """
    SA-MOO adversarial attack algorithm main function.

    Execute complete attack flow:
    1. Load data and model
    2. Initialize population
    3. Evolution loop (initialization->crossover->mutation->selection)
    4. Result analysis and saving
    """
    # Filter warning messages
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision.models._utils")

    # Create run directory and log
    time_str = datetime.now().strftime("%m%d%H%M")
    attack_type = "targeted" if IS_TARGETED_ATTACK else "non_targeted"
    run_name = f"img{TARGET_IMAGE_ID}_{attack_type}_k{FIXED_K}_{PERTURBATION_MODE}_gen{NUM_GENERATIONS}_{time_str}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_dir / "report.txt")
    sys.stdout = logger  # Redirect output to log file

    try:
        # Get compute device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load target image, model and class names
        original_rgb, original_v, true_label, model, class_names = load_target_image_and_model(
            TARGET_IMAGE_ID, MODEL_WEIGHTS_PATH, DATA_ROOT_DIR, PERTURBATION_MODE
        )
        edge_guidance_weights = None
        edge_guidance_status = "DISABLED"
        if EDGE_GUIDANCE_CONFIG.get("enabled", False):
            try:
                edge_guidance_weights = compute_edge_guidance_weights(original_rgb, EDGE_GUIDANCE_CONFIG)
                method = EDGE_GUIDANCE_CONFIG.get("method", "sobel")
                edge_guidance_status = f"ENABLED (method: {method})"
                semantic_cfg = EDGE_GUIDANCE_CONFIG.get("semantic")
                artifacts = getattr(edge_guidance_weights, "artifacts", None)
                if (
                    isinstance(semantic_cfg, dict)
                    and semantic_cfg.get("enabled", False)
                    and artifacts
                ):
                    try:
                        save_edge_guidance_artifacts(
                            artifacts,
                            run_dir,
                            prefix="semantic_guidance",
                        )
                        print("[Edge Guidance] Semantic guidance diagnostics saved.")
                    except Exception as viz_exc:
                        print(f"[Edge Guidance] Failed to save semantic diagnostics: {viz_exc}")
            except Exception as exc:
                print(f"[Edge Guidance] Initialization failed: {exc}. Falling back to uniform sampling.")
                edge_guidance_weights = None

        # Initialize visualization (if enabled)
        if ENABLE_VISUALIZATION:
            init_visualization(original_rgb)

        scheduler = DynamicParameterScheduler(
            DYNAMIC_PARAMETER_CONFIG,
            total_generations=NUM_GENERATIONS,
            is_targeted_attack=IS_TARGETED_ATTACK
        )

        initial_params = scheduler.get_params(0)
        zero_sample_initial = initial_params.get("zero_sample_prob")
        crossover_initial = initial_params.get("crossover_prob")
        fixed_k_initial = initial_params.get("fixed_k")

        current_zero_sample_prob = float(zero_sample_initial if zero_sample_initial is not None else ZERO_SAMPLE_PROB)
        current_crossover_prob = float(crossover_initial if crossover_initial is not None else CROSSOVER_PROB)
        runtime_fixed_k = int(round(fixed_k_initial)) if fixed_k_initial is not None else FIXED_K
        runtime_fixed_k = max(1, runtime_fixed_k)

        fallback_start_pm = 0.4 if not IS_TARGETED_ATTACK else 0.2
        initial_mutation_prob_raw = scheduler.get_value("mutation_probability", 0, fallback_start_pm)
        initial_mutation_prob = float(initial_mutation_prob_raw if initial_mutation_prob_raw is not None else fallback_start_pm)

        mutation_final_raw = scheduler.get_value("mutation_probability", NUM_GENERATIONS - 1, initial_mutation_prob)
        mutation_final = float(mutation_final_raw if mutation_final_raw is not None else initial_mutation_prob)
        crossover_final_raw = scheduler.get_value("crossover_prob", NUM_GENERATIONS - 1, current_crossover_prob)
        crossover_final = float(crossover_final_raw if crossover_final_raw is not None else current_crossover_prob)
        zero_sample_final_raw = scheduler.get_value("zero_sample_prob", NUM_GENERATIONS - 1, current_zero_sample_prob)
        zero_sample_final = float(zero_sample_final_raw if zero_sample_final_raw is not None else current_zero_sample_prob)
        fixed_k_final_raw = scheduler.get_value("fixed_k", NUM_GENERATIONS - 1, runtime_fixed_k)
        fixed_k_final = int(round(fixed_k_final_raw)) if fixed_k_final_raw is not None else runtime_fixed_k
        fixed_k_final = max(1, fixed_k_final)

        def format_range_float(start: float, end: float, precision: int = 3) -> str:
            if abs(end - start) < 1e-9:
                return f"{start:.{precision}f}"
            return f"{start:.{precision}f} -> {end:.{precision}f}"

        def format_range_int(start: int, end: int) -> str:
            if start == end:
                return str(start)
            return f"{start} -> {end}"

        # Initialize population
        population = initialize_population(
            POPULATION_SIZE,
            runtime_fixed_k,
            PERTURBATION_MODE,
            zero_sample_prob=current_zero_sample_prob,
            edge_guidance=edge_guidance_weights
        )

        # Initialize best solution tracking variables
        best_sol = None
        best_obj = (False, float("inf"), float("inf"), float("inf"))

        # History record for convergence curve
        history = {"l2": [], "l0": []}

        # Print attack configuration information
        print("--- Starting Unified SA-MOO Attack ---")
        print(f"Perturbation Mode: {PERTURBATION_MODE}")
        print(f"Attack Type: {'Targeted' if IS_TARGETED_ATTACK else 'Non-Targeted'}")
        if IS_TARGETED_ATTACK:
            print(f"Target Class: {class_names[TARGET_CLASS_ID]} ({TARGET_CLASS_ID})")
        print(f"Fixed K (L0): {format_range_int(runtime_fixed_k, fixed_k_final)}")
        print(f"Generations: {NUM_GENERATIONS}, Population Size: {POPULATION_SIZE}")
        print(f"Crossover Prob: {format_range_float(current_crossover_prob, crossover_final)}")
        print(f"Zero-sample Prob: {format_range_float(current_zero_sample_prob, zero_sample_final)}")
        print(f"Mutation Prob: {format_range_float(initial_mutation_prob, mutation_final)}")
        print(f"Visualization: {'ENABLED' if ENABLE_VISUALIZATION else 'DISABLED'}")
        print(f"Real World Robustness: {'ENABLED' if ENABLE_REAL_WORLD_ROBUSTNESS else 'DISABLED'}")
        if ENABLE_REAL_WORLD_ROBUSTNESS:
            print(f"  JPEG Quality: {JPEG_QUALITY}, Resize: {'ENABLED' if ENABLE_RESIZE_PREPROCESSING else 'DISABLED'}")
            if ENABLE_RESIZE_PREPROCESSING:
                print(f"  Resize Scale: {RESIZE_SCALE}")
        print(f"Edge Guidance: {edge_guidance_status}")

        # ========== Evolution main loop ==========
        for gen in range(NUM_GENERATIONS):
            gen_params = scheduler.get_params(gen)

            pm_override = gen_params.get("mutation_probability")
            if pm_override is not None:
                current_pm = float(pm_override)
            else:
                progress = gen / max(NUM_GENERATIONS - 1, 1)
                current_pm = max(0.001, fallback_start_pm * (1 - progress))

            crossover_override = gen_params.get("crossover_prob")
            if crossover_override is not None:
                current_crossover_prob = float(crossover_override)

            zero_sample_override = gen_params.get("zero_sample_prob")
            if zero_sample_override is not None:
                current_zero_sample_prob = float(zero_sample_override)

            fixed_k_override = gen_params.get("fixed_k")
            if fixed_k_override is not None:
                runtime_fixed_k = max(1, int(round(fixed_k_override)))

            # Generate offspring: crossover operation
            offspring = []
            while len(offspring) < POPULATION_SIZE:
                # Randomly select two parents
                i1, i2 = np.random.choice(POPULATION_SIZE, 2, replace=False)
                p1, p2 = population[i1], population[i2]

                if np.random.rand() < current_crossover_prob:
                    # Perform crossover
                    c1, c2 = crossover(p1, p2, runtime_fixed_k, PERTURBATION_MODE, current_crossover_prob)
                    offspring.extend([c1, c2])
                else:
                    # No crossover, copy directly
                    offspring.extend([p1, p2])

            offspring = offspring[:POPULATION_SIZE]  # Ensure correct offspring count

            # Mutation operation
            mutated_offspring = [
                mutate(
                    ind,
                    runtime_fixed_k,
                    current_pm,
                    PERTURBATION_MODE,
                    current_zero_sample_prob,
                    edge_guidance=edge_guidance_weights,
                )
                for ind in offspring
            ]

            # Combine parent and offspring, perform batch evaluation
            combined = population + mutated_offspring
            obj_vals = evaluate_objectives_batch(
                combined, original_rgb, original_v, true_label, model,
                IS_TARGETED_ATTACK, TARGET_CLASS_ID if IS_TARGETED_ATTACK else None,
                PERTURBATION_MODE
            )

            # Select next generation population
            population = selection(combined, obj_vals, POPULATION_SIZE)

            # Update best solution
            current_obj = evaluate_objectives_batch(
                population, original_rgb, original_v, true_label, model,
                IS_TARGETED_ATTACK, TARGET_CLASS_ID if IS_TARGETED_ATTACK else None,
                PERTURBATION_MODE
            )

            # Select best individual from first front
            fronts = non_dominated_sort(current_obj)
            best_idx = fronts[0][0] if fronts[0] else 0

            if dominates(current_obj[best_idx], best_obj):
                best_sol = population[best_idx]
                best_obj = current_obj[best_idx]

            # Record history (only when attack succeeds)
            if best_obj[0]:
                history["l2"].append(best_obj[2])
                history["l0"].append(best_obj[3])
            else:
                history["l2"].append(None)
                history["l0"].append(None)

            # Print current generation information
            print(
                f"Gen {gen+1}/{NUM_GENERATIONS} | PM: {current_pm:.3f} | XO: {current_crossover_prob:.3f} | "
                f"ZP: {current_zero_sample_prob:.3f} | Best L2: {best_obj[2]:.2f}, "
                f"L0: {best_obj[3]}, Success: {best_obj[0]}"
            )

            # Update visualization
            if ENABLE_VISUALIZATION and (gen + 1) % VISUALIZE_INTERVAL == 0:
                update_visualization(
                    original_rgb, best_sol, best_obj, gen + 1,
                    PERTURBATION_MODE, original_v
                )

        # Close visualization
        if ENABLE_VISUALIZATION:
            plt.ioff()
            plt.show()

        # ========== Final result analysis ==========
        print("\n--- Attack Completed ---")

        if best_sol is None:
            print("Failed to find an adversarial example.")
            return

        # Reconstruct final perturbed image
        indices, perturbations = best_sol
        if PERTURBATION_MODE == "v_channel":
            noise_full = np.zeros_like(original_v, dtype=np.float32)
            noise_full.flat[indices] = perturbations
            original_hsv = rgb_to_hsv(original_rgb)
            final_v = np.clip(original_hsv[:, :, 2] + noise_full, 0.0, 1.0)
            final_hsv = np.stack([
                original_hsv[:, :, 0],
                original_hsv[:, :, 1],
                final_v
            ], axis=-1)
            final_rgb = hsv_to_rgb(final_hsv)
        else:
            noise_full = np.zeros_like(original_rgb, dtype=np.float32)
            noise_full.flat[indices] = perturbations
            final_rgb = np.clip(original_rgb + noise_full, 0.0, 1.0)

        # Calculate quality metrics
        psnr_value = psnr(original_rgb, final_rgb, data_range=1.0)

        print("Final Best Solution Stats:")
        print(f"  Adversarial: {best_obj[0]}")
        print(f"  L2 Norm: {best_obj[2]:.4f}")
        print(f"  L0 Norm: {best_obj[3]}")
        print(f"  PSNR: {psnr_value:.2f} dB")

        # LPIPS perceptual loss
        loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
        orig_tensor = torch.from_numpy(original_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
        pert_tensor = torch.from_numpy(final_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
        with torch.no_grad():
            lpips_val = float(loss_fn_vgg(orig_tensor, pert_tensor).item())
        print(f"  LPIPS: {lpips_val:.4f}")

        # CIELAB color space analysis
        original_lab = rgb2lab(original_rgb)
        perturbed_lab = rgb2lab(final_rgb)
        delta_E00 = deltaE_ciede2000(original_lab, perturbed_lab)
        mean_delta_E00 = float(np.mean(delta_E00))
        max_delta_E00 = float(np.max(delta_E00))
        print("\n--- CIELAB Analysis ---")
        print(f"  Mean ΔE00: {mean_delta_E00:.3f}")
        print(f"  Max ΔE00: {max_delta_E00:.3f}")

        # Save ΔL* heatmap
        delta_L = perturbed_lab[:, :, 0] - original_lab[:, :, 0]
        save_delta_L_heatmap(delta_L, run_dir / "deltaL_heatmap.png")

        # Final prediction verification
        with torch.no_grad():
            # If real world robustness is enabled, also apply preprocessing to final image
            verification_rgb = final_rgb
            if ENABLE_REAL_WORLD_ROBUSTNESS:
                verification_rgb = apply_real_world_preprocessing(final_rgb)
            
            final_tensor = torch.from_numpy(verification_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
            output_logits = model(final_tensor)
            predicted_class_idx = int(torch.argmax(output_logits, dim=1).item())
            predicted_class_name = class_names[predicted_class_idx]

        print("\n--- Prediction Verification ---")
        if ENABLE_REAL_WORLD_ROBUSTNESS:
            print(f"Note: Verification applied real-world preprocessing (JPEG quality: {JPEG_QUALITY}, resize: {'enabled' if ENABLE_RESIZE_PREPROCESSING else 'disabled'})")
        else:
            print("Note: Verification without preprocessing")

        # Original image prediction
        orig_tensor = torch.from_numpy(original_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            orig_logits = model(orig_tensor)
            orig_probs = torch.softmax(orig_logits, dim=1).cpu().squeeze()

        pert_probs = torch.softmax(output_logits, dim=1).cpu().squeeze()
        true_class_name = class_names[true_label]
        orig_true_prob = float(orig_probs[true_label].item() * 100)
        pert_true_prob = float(pert_probs[true_label].item() * 100)
        pert_max_prob = float(pert_probs[predicted_class_idx].item() * 100)

        # Print different information according to attack type
        if IS_TARGETED_ATTACK:
            target_class_name = class_names[TARGET_CLASS_ID]
            orig_target_prob = float(orig_probs[TARGET_CLASS_ID].item() * 100)
            pert_target_prob = float(pert_probs[TARGET_CLASS_ID].item() * 100)
            print(f"Original class: {true_class_name} ({true_label})")
            print(f"Target class: {target_class_name} ({TARGET_CLASS_ID})")
            print(f"Predicted class after attack: {predicted_class_name} ({predicted_class_idx})")
            print(f"Attack Success: {predicted_class_idx == TARGET_CLASS_ID}")
            print(f"Original class probability dropped from {orig_true_prob:.2f}% to {pert_true_prob:.2f}%")
            print(f"Target class probability increased from {orig_target_prob:.2f}% to {pert_target_prob:.2f}%")
            if predicted_class_idx == TARGET_CLASS_ID:
                print(f"Model now confidently predicts the target '{target_class_name}' with {pert_max_prob:.2f}% confidence.")
            else:
                print(f"Model failed to predict the target. It predicts '{predicted_class_name}' with {pert_max_prob:.2f}% confidence.")
        else:
            print(f"Original class: {true_class_name} ({true_label})")
            print(f"Predicted class after attack: {predicted_class_name} ({predicted_class_idx})")
            print(f"Attack Success: {predicted_class_idx != true_label}")
            print(f"Original class probability dropped from {orig_true_prob:.2f}% to {pert_true_prob:.2f}%")
            print(f"Model now confidently predicts '{predicted_class_name}' with {pert_max_prob:.2f}% confidence.")

        # Save noise heatmap
        if PERTURBATION_MODE == "v_channel":
            save_noise_heatmap(noise_full, "Value", run_dir / "noise_V.png")
        else:
            noise_l2 = np.linalg.norm(noise_full, axis=2)
            save_noise_heatmap(noise_l2, "RGB L2", run_dir / "noise_RGB_L2.png")

        # Save final result plot
        if PERTURBATION_MODE == "v_channel":
            noise_display = noise_full.copy()
        else:
            noise_display = noise_l2.copy()

        if noise_display.max() > noise_display.min():
            noise_display = (noise_display - noise_display.min()) / (noise_display.max() - noise_display.min())

        save_adversarial_result(
            original_rgb, noise_display, final_rgb,
            true_class_name, predicted_class_name, best_obj[3],
            PERTURBATION_MODE, run_dir / "adversarial_result.png"
        )

        # Save 32x32 perturbed image
        final_img_only_path = run_dir / "final_perturbed.png"
        final_perturbed_uint8 = (final_rgb * 255).astype(np.uint8)
        img_pil = Image.fromarray(final_perturbed_uint8)
        img_pil.save(final_img_only_path, format="PNG", compress_level=0)

        # If real world robustness is enabled, save preprocessed image
        if ENABLE_REAL_WORLD_ROBUSTNESS:
            preprocessed_rgb = apply_real_world_preprocessing(final_rgb)
            preprocessed_uint8 = (preprocessed_rgb * 255).astype(np.uint8)
            preprocessed_path = run_dir / "final_perturbed_preprocessed.png"
            preprocessed_pil = Image.fromarray(preprocessed_uint8)
            preprocessed_pil.save(preprocessed_path, format="PNG", compress_level=0)
            print(f"Preprocessed image saved as: final_perturbed_preprocessed.png")

        # Save convergence curve
        save_convergence_plot(
            history["l2"], history["l0"], run_dir / "convergence.png"
        )

        print(f"\nResults saved in: {run_dir.absolute()}")
        print(f"Noise heatmap saved as: {'noise_V.png' if PERTURBATION_MODE == 'v_channel' else 'noise_RGB_L2.png'}")
        print(f"Final 32x32 perturbed image saved as: {final_img_only_path.name}")

    finally:
        # Restore original stdout and close log
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
