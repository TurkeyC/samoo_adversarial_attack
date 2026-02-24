"""
Configuration manager for SA-MOO
Handles loading and managing configurations from various sources
"""

import os
import sys
import json
import copy
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
from dotenv import load_dotenv

from .types import (
    SystemConfig,
    ModelConfig,
    SaMooConfig,
    AttackConfig,
    ContinuousPerturbationConfig,
    VisualizationConfig,
    RealWorldRobustnessConfig,
    DominanceConfig,
    DynamicParameterConfig,
    EdgeGuidanceConfig
)


class ConfigManager:
    """
    Configuration Manager for SA-MOO
    
    Handles loading and managing configurations from multiple sources with priority:
    1. Command line arguments (highest priority)
    2. Environment variables
    3. YAML configuration file
    4. Default values (lowest priority)
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: YAML configuration file path, if None uses default search
        """
        load_dotenv()
        
        self.config_file = config_file or self._find_config_file()
        self.yaml_config = self._load_yaml_config()
        self.args = self._parse_args()
        
        # Build the complete configuration
        self.system_config = self._build_system_config()

    def _find_config_file(self) -> Optional[str]:
        """Find configuration file"""
        # First check environment variable
        config_path = os.getenv("CONFIG_FILE")
        if config_path and Path(config_path).exists():
            return config_path

        # Then check default locations
        default_configs = ["config.yaml", "config.yml", "../config.yaml", "../config.yml"]
        for config_name in default_configs:
            config_path = Path(__file__).parent.parent / config_name
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
        parser.add_argument("--targeted", action="store_true", help="Whether to perform targeted attack")
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
        parser.add_argument("--dominance-config", type=str, help="Dynamic dominance relationship config (JSON string or file path)")
        parser.add_argument("--dynamic-params", type=str, help="Dynamic parameter adjustment config (JSON string or file path)")
        parser.add_argument("--edge-guidance", type=str, help="Edge guidance config (JSON string or file path)")

        # Other
        parser.add_argument("--config", type=str, help="Specify config file path")

        return parser.parse_args()

    def _get_config_value(self, key: str, default_value: Any, value_type: type = str) -> Any:
        """
        Get configuration value from multiple sources with priority
        
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
        """Parse external config that can be JSON string or file path."""
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
        """Deep merge config dictionaries, not modifying input dicts."""
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

    def _load_dominance_config(self) -> DominanceConfig:
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

        return DominanceConfig(
            rules=dominance_config["rules"],
            tie_breakers=dominance_config["tie_breakers"]
        )

    def _load_dynamic_parameters(self) -> DynamicParameterConfig:
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
                "value": self._get_config_value("crossover_prob", 0.1, float)
            },
            "zero_sample_prob": {
                "schedule": "constant",
                "value": self._get_config_value("zero_sample_prob", 0.3, float)
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

        return DynamicParameterConfig(
            mutation_probability=dynamic_config["mutation_probability"],
            crossover_prob=dynamic_config["crossover_prob"],
            zero_sample_prob=dynamic_config["zero_sample_prob"]
        )

    def _load_edge_guidance_config(self) -> EdgeGuidanceConfig:
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

        return EdgeGuidanceConfig(
            enabled=edge_config["enabled"],
            method=edge_config["method"],
            gaussian_sigma=edge_config["gaussian_sigma"],
            canny_sigma=edge_config["canny_sigma"],
            exponent=edge_config["exponent"],
            uniform_mix=edge_config["uniform_mix"],
            min_value=edge_config["min_value"],
            multi_scale=edge_config["multi_scale"],
            semantic=edge_config["semantic"],
            professional_preprocessing=edge_config["professional_preprocessing"]
        )

    def _build_system_config(self) -> SystemConfig:
        """Build the complete system configuration"""
        # Load all sub-configurations
        model_config = ModelConfig(
            weights_path=self._get_config_value("model_weights_path", "cifar10_resnet18.pth", str),
            data_root_dir=self._get_config_value("data_root_dir", "./data", str),
            output_dir=self._get_config_value("output_dir", "../output", str),
            torch_home=os.getenv("TORCH_HOME"),
            lpips_cache_dir=os.getenv("LPIPS_CACHE_DIR")
        )
        
        # Convert string paths to Path objects
        model_config.weights_path = Path(model_config.weights_path)
        model_config.data_root_dir = Path(model_config.data_root_dir)
        model_config.output_dir = Path(model_config.output_dir)
        
        # Check if model weights exist
        if not model_config.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_config.weights_path.absolute()}.")
        
        # Create output directory if it doesn't exist
        model_config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for PyTorch and LPIPS
        torch_home = model_config.torch_home or Path.home() / ".cache" / "torch"
        lpips_cache = model_config.lpips_cache_dir or Path.home() / ".cache" / "torch" / "lpips"
        
        os.environ["TORCH_HOME"] = str(torch_home)
        os.environ["LPIPS_CACHE_DIR"] = str(lpips_cache)

        sa_moo_config = SaMooConfig(
            population_size=self._get_config_value("population_size", 2, int),
            num_generations=self._get_config_value("num_generations", 1000, int),
            fixed_k=self._get_config_value("fixed_k", 24, int),
            crossover_prob=self._get_config_value("crossover_prob", 0.1, float),
            zero_sample_prob=self._get_config_value("zero_sample_prob", 0.3, float)
        )

        attack_config = AttackConfig(
            perturbation_mode=self._get_config_value("perturbation_mode", "rgb_sim", str),
            is_targeted_attack=self._get_config_value("is_targeted_attack", False, bool),
            target_class_id=self._get_config_value("target_class_id", 5, int) if self._get_config_value("is_targeted_attack", False, bool) else None
        )

        continuous_perturbation_config = ContinuousPerturbationConfig(
            enable=self._get_config_value("enable_continuous_perturbation", False, bool),
            lower_bound=self._get_config_value("continuous_lower_bound", -1.0, float),
            upper_bound=self._get_config_value("continuous_upper_bound", 1.0, float),
            decimal_places=self._get_config_value("continuous_decimal_places", 2, int)
        )

        visualization_config = VisualizationConfig(
            enable=self._get_config_value("enable_visualization", False, bool),
            interval=self._get_config_value("visualize_interval", 500, int)
        )

        real_world_robustness_config = RealWorldRobustnessConfig(
            enable=self._get_config_value("enable_real_world_robustness", False, bool),
            jpeg_quality=self._get_config_value("jpeg_quality", 75, int),
            enable_resize_preprocessing=self._get_config_value("enable_resize_preprocessing", True, bool),
            resize_scale=self._get_config_value("resize_scale", 2.0, float)
        )

        return SystemConfig(
            target_image_id=self._get_config_value("target_image_id", 7780, int),
            model=model_config,
            sa_moo=sa_moo_config,
            attack=attack_config,
            continuous_perturbation=continuous_perturbation_config,
            visualization=visualization_config,
            real_world_robustness=real_world_robustness_config,
            dominance=self._load_dominance_config(),
            dynamic_parameters=self._load_dynamic_parameters(),
            edge_guidance=self._load_edge_guidance_config()
        )

    def print_config(self):
        """Print current configuration"""
        cfg = self.system_config
        print("=== Current Configuration ===")
        print(f"Target Image ID: {cfg.target_image_id}")
        print(f"Model Weights: {cfg.model.weights_path}")
        print(f"Data Root: {cfg.model.data_root_dir}")
        print(f"Output Dir: {cfg.model.output_dir}")
        print(f"Perturbation Mode: {cfg.attack.perturbation_mode}")
        print(f"Targeted Attack: {cfg.attack.is_targeted_attack}")
        if cfg.attack.is_targeted_attack:
            print(f"Target Class ID: {cfg.attack.target_class_id}")
        print(f"Continuous Perturbation: {cfg.continuous_perturbation.enable}")
        if cfg.continuous_perturbation.enable:
            print(f"Continuous Bounds: [{cfg.continuous_perturbation.lower_bound}, {cfg.continuous_perturbation.upper_bound}]")
            print(f"Decimal Places: {cfg.continuous_perturbation.decimal_places}")
        print(f"Population Size: {cfg.sa_moo.population_size}")
        print(f"Generations: {cfg.sa_moo.num_generations}")
        print(f"Fixed K: {cfg.sa_moo.fixed_k}")
        print(f"Visualization: {cfg.visualization.enable}")
        print(f"Real World Robustness: {cfg.real_world_robustness.enable}")
        print("=" * 30)