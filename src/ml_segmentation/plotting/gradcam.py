"""
GradCAM (Gradient-weighted Class Activation Mapping) for model explainability.

Visualizes which regions of the input image the model focuses on when making predictions.
Useful for understanding why the model predicts thick joints or misses joints in fractured areas.

References:
    Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" https://arxiv.org/abs/1610.02391
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

# Optional dependency: OpenCV (cv2)
# If not available, GradCAM visualization will be skipped
try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None  # type: ignore


class GradCAM:
    """
    Generate GradCAM heatmaps for segmentation models.

    Highlights which features/regions the model uses to make predictions.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize GradCAM.

        Args:
            model: Trained segmentation model (e.g., U-Net)
            target_layer: Layer to visualize (typically last encoder layer)
                         For ResNet encoder: model.encoder.layer4
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        # Register hooks to capture activations and gradients
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(
        self,
        module: torch.nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Hook to save forward pass activations."""
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        """Hook to save backward pass gradients."""
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def generate_cam(
        self, input_image: torch.Tensor, threshold: float = 0.3
    ) -> NDArray[np.float32]:
        """
        Generate GradCAM heatmap for input image.

        Args:
            input_image: Input tensor (B, C, H, W), typically B=1
            threshold: Prediction threshold for generating target signal

        Returns:
            Normalized heatmap as numpy array (H, W) with values in [0, 1]
        """
        # Set to eval mode but ensure gradients are enabled for GradCAM
        self.model.eval()

        # Ensure model parameters require gradients (critical for backward pass)
        for param in self.model.parameters():
            param.requires_grad = True

        # Ensure input requires gradients
        input_image = input_image.requires_grad_(True)

        # Forward pass (must be inside torch.enable_grad())
        with torch.enable_grad():
            output = self.model(input_image)

            # For segmentation: compute scalar target for backprop
            # Use output directly to maintain gradient flow
            # Option 1: Use mean of all outputs (simple and reliable)
            # Option 2: Use weighted sum based on predictions
            # We use a differentiable approximation instead of hard threshold

            # Soft target: use sigmoid to create smooth weights instead of hard threshold
            # This maintains gradient flow unlike (output > threshold).float()
            target = output.sum()

            # Alternative: if we want to focus on high-confidence predictions:
            # target = (output * torch.sigmoid(10 * (output - threshold))).sum()

            # Backward pass to get gradients
            self.model.zero_grad()
            target.backward()

        # Check if gradients were captured
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradients or activations not captured. "
                "Check target_layer registration."
            )

        # Weight activations by gradient importance
        # Average gradients across spatial dimensions (global average pooling)
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weight each activation map by its importance
        weighted_activations = pooled_gradients * self.activations

        # Sum across channels to get single heatmap
        cam = torch.sum(weighted_activations, dim=1, keepdim=True)

        # Apply ReLU to only keep positive influences
        cam = F.relu(cam)

        # Upsample to input image size
        cam = F.interpolate(
            cam, size=input_image.shape[2:], mode="bilinear", align_corners=False
        )

        # Normalize to [0, 1]
        cam_np = cam.squeeze().cpu().numpy()
        cam_np = cam_np - cam_np.min()
        cam_max = cam_np.max()
        if cam_max > 0:
            cam_np = cam_np / cam_max

        return cam_np.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove registered hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()

    def __del__(self):
        """Cleanup hooks on deletion."""
        try:
            self.remove_hooks()
        except (AttributeError, RuntimeError):
            pass


def overlay_heatmap(
    image: NDArray[np.uint8],
    heatmap: NDArray[np.float32],
    alpha: float = 0.4,
    colormap: int | None = None,
) -> NDArray[np.uint8]:
    """
    Overlay GradCAM heatmap on original image.

    Args:
        image: Original RGB image (H, W, 3) in range [0, 255]
        heatmap: GradCAM heatmap (H, W) in range [0, 1]
        alpha: Blending factor (0=only image, 1=only heatmap)
        colormap: OpenCV colormap (default: JET = red=high, blue=low)
                 Only used if OpenCV is available

    Returns:
        Overlayed image (H, W, 3) in range [0, 255]
    """
    if not HAS_OPENCV:
        # Fallback: simple matplotlib-based colormap if OpenCV not available
        import matplotlib.cm as cm

        heatmap_colored = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    else:
        # Convert heatmap to RGB using OpenCV colormap
        if colormap is None:
            colormap = cv2.COLORMAP_JET  # type: ignore
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)  # type: ignore
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # type: ignore

    # Blend image and heatmap
    overlayed = (1 - alpha) * image + alpha * heatmap_colored
    return overlayed.astype(np.uint8)


def generate_gradcam_visualizations(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: Path,
    num_samples: int = 10,
    threshold: float = 0.3,
) -> None:
    """
    Generate and save GradCAM visualizations for validation samples.

    Creates side-by-side comparison: Original | GradCAM | Prediction | Ground Truth

    Args:
        model: Trained segmentation model
        dataloader: Validation/test dataloader
        device: Device to run on
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        threshold: Prediction threshold
    """
    if not HAS_OPENCV:
        print(
            "Warning: OpenCV (cv2) not available. GradCAM will use matplotlib "
            "colormap fallback (slightly different colors than OpenCV JET)."
        )

    # Identify target layer (last encoder layer for U-Net with ResNet)
    target_layer = _get_target_layer(model)

    if target_layer is None:
        print("Warning: Could not identify target layer for GradCAM. Skipping.")
        return

    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer)

    # Create output directory
    gradcam_dir = save_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    samples_processed = 0

    for _batch_idx, batch in enumerate(dataloader):
        if samples_processed >= num_samples:
            break

        # Handle both (images, masks) and dict formats
        if isinstance(batch, dict):
            images = batch["image"]
            masks = batch["mask"]
        else:
            images, masks = batch

        images = images.to(device)
        masks = masks.cpu()

        for i in range(images.shape[0]):
            if samples_processed >= num_samples:
                break

            # Process single image
            single_image = images[i : i + 1]

            # Generate GradCAM heatmap (handles gradient enabling internally)
            cam = gradcam.generate_cam(single_image, threshold=threshold)

            # Generate prediction
            with torch.no_grad():
                pred = model(single_image)
            pred_mask = (pred > threshold).float().cpu()

            # Prepare original image for visualization
            img_np = single_image[0].cpu().permute(1, 2, 0).numpy()
            # Normalize to [0, 1]
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            img_np = (img_np * 255).astype(np.uint8)

            # Create overlay
            overlay = overlay_heatmap(img_np, cam)

            # Create 4-panel figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(img_np)
            axes[0].set_title("Original Image", fontsize=14)
            axes[0].axis("off")

            axes[1].imshow(overlay)
            axes[1].set_title("GradCAM Overlay", fontsize=14)
            axes[1].axis("off")

            axes[2].imshow(pred_mask[0, 0], cmap="gray", vmin=0, vmax=1)
            axes[2].set_title(f"Prediction (threshold={threshold})", fontsize=14)
            axes[2].axis("off")

            axes[3].imshow(masks[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[3].set_title("Ground Truth", fontsize=14)
            axes[3].axis("off")

            plt.tight_layout()
            output_path = gradcam_dir / f"gradcam_sample_{samples_processed:03d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            samples_processed += 1

    # Cleanup
    gradcam.remove_hooks()

    print(f"Generated {samples_processed} GradCAM visualizations in {gradcam_dir}")


def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    """
    Automatically identify the target layer for GradCAM.

    For U-Net with ResNet encoder: returns encoder.layer4
    For other architectures: attempts to find last convolutional block

    Args:
        model: Segmentation model

    Returns:
        Target layer or None if not found
    """
    # Try common encoder structures
    if hasattr(model, "encoder"):
        # ResNet-based encoder (most common)
        if hasattr(model.encoder, "layer4"):
            return model.encoder.layer4
        elif hasattr(model.encoder, "stages"):
            # EfficientNet-based encoder
            return model.encoder.stages[-1]
        elif hasattr(model.encoder, "layers"):
            return model.encoder.layers[-1]

    # Fallback: find last convolutional block
    conv_modules = [
        m
        for m in model.modules()
        if isinstance(m, torch.nn.Conv2d | torch.nn.Sequential)
    ]
    if conv_modules:
        # Return a module near the end of encoder (heuristic: ~70% through)
        target_idx = int(len(conv_modules) * 0.7)
        return conv_modules[target_idx]

    return None
