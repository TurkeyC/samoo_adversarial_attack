"""
Visualizer for SA-MOO
Handles visualization and result saving
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional

from ..config.types import SystemConfig


class Visualizer:
    """
    Visualizer for SA-MOO
    
    Handles visualization during the evolutionary process and saves results
    including noise heatmaps, convergence plots, and final adversarial examples.
    """
    
    def __init__(self, original_rgb: np.ndarray):
        """
        Initialize the visualizer with the original image
        
        Args:
            original_rgb: Original RGB image
        """
        self.original_rgb = original_rgb
        self.fig = None
        self.axs = None
        
    def init_visualization(self):
        """Initialize visualization windows"""
        # Create a figure with subplots
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))
        plt.ion()  # Turn on interactive mode
        
    def update(
        self, 
        original_rgb: np.ndarray, 
        best_solution: Optional[Tuple[np.ndarray, np.ndarray]], 
        best_objectives: Tuple[bool, float, float, float], 
        generation: int,
        perturbation_mode: str,
        original_v: Optional[np.ndarray] = None
    ):
        """
        Update visualization with current best solution
        
        Args:
            original_rgb: Original RGB image
            best_solution: Current best solution (indices, perturbations)
            best_objectives: Current best objectives
            generation: Current generation number
            perturbation_mode: Perturbation mode
            original_v: Original V channel (for V-channel mode)
        """
        if self.fig is None:
            self.init_visualization()
            
        # Clear the axes
        for ax in self.axs:
            ax.clear()
            
        if best_solution is not None:
            # Reconstruct perturbed image
            indices, perturbations = best_solution
            
            if perturbation_mode == "v_channel" and original_v is not None:
                from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
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
                
            # Plot original, perturbed, and noise
            self.axs[0].imshow(original_rgb)
            self.axs[0].set_title("Original")
            self.axs[0].axis('off')
            
            self.axs[1].imshow(final_rgb)
            self.axs[1].set_title(f"Generation {generation}\nPerturbed (Success: {best_objectives[0]})")
            self.axs[1].axis('off')
            
            # Show noise as heatmap
            if perturbation_mode == "v_channel":
                noise_display = noise_full
            else:
                noise_display = np.linalg.norm(noise_full, axis=2)
                
            im = self.axs[2].imshow(noise_display, cmap='RdBu_r', vmin=-1, vmax=1)
            self.axs[2].set_title("Noise Heatmap")
            self.axs[2].axis('off')
            plt.colorbar(im, ax=self.axs[2])
            
        else:
            # Just show original
            self.axs[0].imshow(original_rgb)
            self.axs[0].set_title("Original")
            self.axs[0].axis('off')
            
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot
        
    def save_results(
        self,
        original_rgb: np.ndarray,
        noise_full: np.ndarray,
        final_rgb: np.ndarray,
        true_label: int,
        model: torch.nn.Module,
        class_names: List[str],
        best_objectives: Tuple[bool, float, float, float],
        perturbation_mode: str,
        run_dir: Path
    ):
        """
        Save final results including images, metrics, and heatmaps
        
        Args:
            original_rgb: Original RGB image
            noise_full: Full noise array
            final_rgb: Final perturbed RGB image
            true_label: True label of the original image
            model: Target model
            class_names: List of class names
            best_objectives: Final best objectives
            perturbation_mode: Perturbation mode
            run_dir: Directory to save results
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate quality metrics
        psnr_value = psnr(original_rgb, final_rgb, data_range=1.0)
        
        # LPIPS perceptual loss
        orig_tensor = torch.from_numpy(original_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
        pert_tensor = torch.from_numpy(final_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
        
        loss_fn_vgg = torch.nn.Module()  # Placeholder since we're not using the full model here
        import lpips
        loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
        
        with torch.no_grad():
            lpips_val = float(loss_fn_vgg(orig_tensor, pert_tensor).item())
            
        # CIELAB color space analysis
        original_lab = rgb2lab(original_rgb)
        perturbed_lab = rgb2lab(final_rgb)
        delta_E00 = deltaE_ciede2000(original_lab, perturbed_lab)
        mean_delta_E00 = float(np.mean(delta_E00))
        max_delta_E00 = float(np.max(delta_E00))
        
        # Final prediction verification
        with torch.no_grad():
            final_tensor = torch.from_numpy(final_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
            output_logits = model(final_tensor)
            predicted_class_idx = int(torch.argmax(output_logits, dim=1).item())
            predicted_class_name = class_names[predicted_class_idx]
            
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
        
        # Save noise heatmap
        if perturbation_mode == "v_channel":
            self._save_noise_heatmap(noise_full, "Value", run_dir / "noise_V.png")
        else:
            noise_l2 = np.linalg.norm(noise_full, axis=2)
            self._save_noise_heatmap(noise_l2, "RGB L2", run_dir / "noise_RGB_L2.png")
            
        # Save ΔL* heatmap
        delta_L = perturbed_lab[:, :, 0] - original_lab[:, :, 0]
        self._save_delta_L_heatmap(delta_L, run_dir / "deltaL_heatmap.png")
        
        # Save adversarial result
        if perturbation_mode == "v_channel":
            noise_display = noise_full.copy()
        else:
            noise_display = noise_l2.copy()
            
        if noise_display.max() > noise_display.min():
            noise_display = (noise_display - noise_display.min()) / (noise_display.max() - noise_display.min())
            
        self._save_adversarial_result(
            original_rgb, noise_display, final_rgb,
            true_class_name, predicted_class_name, best_objectives[3],
            perturbation_mode, run_dir / "adversarial_result.png"
        )
        
        # Save final 32x32 perturbed image
        final_img_only_path = run_dir / "final_perturbed.png"
        final_perturbed_uint8 = (final_rgb * 255).astype(np.uint8)
        img_pil = Image.fromarray(final_perturbed_uint8)
        img_pil.save(final_img_only_path, format="PNG", compress_level=0)
        
        # Print summary
        print("Final Best Solution Stats:")
        print(f"  Adversarial: {best_objectives[0]}")
        print(f"  L2 Norm: {best_objectives[2]:.4f}")
        print(f"  L0 Norm: {best_objectives[3]}")
        print(f"  PSNR: {psnr_value:.2f} dB")
        print(f"  LPIPS: {lpips_val:.4f}")
        print(f"  Mean ΔE00: {mean_delta_E00:.3f}")
        print(f"  Max ΔE00: {max_delta_E00:.3f}")
        print(f"  Original class: {true_class_name} ({true_label})")
        print(f"  Predicted class after attack: {predicted_class_name} ({predicted_class_idx})")
        print(f"  Original class probability dropped from {orig_true_prob:.2f}% to {pert_true_prob:.2f}%")
        print(f"  Model now predicts '{predicted_class_name}' with {pert_max_prob:.2f}% confidence.")
        
    def _save_noise_heatmap(self, noise: np.ndarray, title: str, save_path: Path):
        """
        Save noise heatmap
        
        Args:
            noise: Noise array
            title: Title for the plot
            save_path: Path to save the image
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(noise, cmap='RdBu_r', interpolation='nearest')
        plt.title(f'{title} Noise Heatmap')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
    def _save_delta_L_heatmap(self, delta_L: np.ndarray, save_path: Path):
        """
        Save ΔL* heatmap
        
        Args:
            delta_L: ΔL* array
            save_path: Path to save the image
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(delta_L, cmap='RdBu_r', interpolation='nearest')
        plt.title('ΔL* Heatmap (Lightness Difference)')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
    def _save_adversarial_result(
        self,
        original_rgb: np.ndarray,
        noise_display: np.ndarray,
        final_rgb: np.ndarray,
        true_class_name: str,
        predicted_class_name: str,
        l0_norm: int,
        perturbation_mode: str,
        save_path: Path
    ):
        """
        Save adversarial result comparison
        
        Args:
            original_rgb: Original RGB image
            noise_display: Noise display array
            final_rgb: Final perturbed RGB image
            true_class_name: True class name
            predicted_class_name: Predicted class name after attack
            l0_norm: L0 norm value
            perturbation_mode: Perturbation mode
            save_path: Path to save the image
        """
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axs[0].imshow(original_rgb)
        axs[0].set_title(f'Original\n({true_class_name})')
        axs[0].axis('off')
        
        # Noise
        axs[1].imshow(noise_display, cmap='RdBu_r', vmin=0, vmax=1)
        axs[1].set_title(f'Noise (L0={l0_norm})\n({perturbation_mode})')
        axs[1].axis('off')
        
        # Perturbed image
        axs[2].imshow(final_rgb)
        axs[2].set_title(f'Adversarial\n→ {predicted_class_name}')
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
    def save_convergence_plot(self, l2_history: List[Optional[float]], l0_history: List[Optional[float]], save_path: Path):
        """
        Save convergence plot
        
        Args:
            l2_history: History of L2 norm values
            l0_history: History of L0 norm values
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        generations = range(1, len(l2_history) + 1)
        
        # Plot L2 norm
        valid_l2 = [(i, val) for i, val in enumerate(l2_history, 1) if val is not None]
        if valid_l2:
            gen_l2, vals_l2 = zip(*valid_l2)
            plt.plot(gen_l2, vals_l2, label='L2 Norm', marker='o', markersize=3, linewidth=1)
        
        # Plot L0 norm
        valid_l0 = [(i, val) for i, val in enumerate(l0_history, 1) if val is not None]
        if valid_l0:
            gen_l0, vals_l0 = zip(*valid_l0)
            plt.plot(gen_l0, vals_l0, label='L0 Norm', marker='s', markersize=3, linewidth=1)
        
        plt.xlabel('Generation')
        plt.ylabel('Norm Value')
        plt.title('Convergence Plot: L2 and L0 Norms over Generations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
    def cleanup(self):
        """Clean up visualization resources"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.axs = None