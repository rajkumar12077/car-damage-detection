import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAMDamageAnalyzer:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.zero_grad()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        car_logits, damage_logits, damage_percent = self.model(input_tensor)

        if class_idx is None:
            class_idx = damage_logits.argmax(dim=1).item()

        score = damage_logits[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.model.gradients
        feature_maps = self.model.feature_maps

        if gradients is None or feature_maps is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def calculate_damage_percentage(self, heatmap: np.ndarray, damage_class_name: str, regression_percent: float = None, threshold: float = 0.35):
        if damage_class_name.lower() in ["no damage", "undamaged", "clean", "none"]:
            return {
                "area_percent": 0.0,
                "regression_percent": 0.0 if regression_percent is None else round(regression_percent, 1),
                "final_percent": 0.0,
                "severity": "None",
                "mask": np.zeros_like(heatmap, dtype=np.uint8)
            }

        binary_mask = (heatmap >= threshold).astype(np.uint8)
        damaged_pixels = np.count_nonzero(binary_mask)
        total_pixels = heatmap.size

        area_pct = (damaged_pixels / total_pixels) * 100.0

        weighted_intensity = np.sum(heatmap * binary_mask) / (damaged_pixels + 1e-6)
        adjusted_area_pct = area_pct * (0.8 + 0.4 * weighted_intensity)
        adjusted_area_pct = min(100.0, max(1.0, adjusted_area_pct))

        if regression_percent is not None and regression_percent > 0:
            final_pct = 0.6 * adjusted_area_pct + 0.4 * regression_percent
        else:
            final_pct = adjusted_area_pct

        final_pct = round(min(100.0, max(0.0, float(final_pct))), 1)

        if final_pct < 5.0:
            severity = "Negligible / Superficial"
        elif final_pct < 20.0:
            severity = "Minor"
        elif final_pct < 50.0:
            severity = "Moderate"
        else:
            severity = "Severe / Critical"

        return {
            "area_percent": round(float(adjusted_area_pct), 1),
            "regression_percent": round(float(regression_percent), 1) if regression_percent is not None else None,
            "final_percent": final_pct,
            "severity": severity,
            "mask": binary_mask
        }

    def overlay_heatmap_on_image(self, original_pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5):
        orig_np = np.array(original_pil_img.convert("RGB"))
        h, w, _ = orig_np.shape

        resized_heatmap = cv2.resize(heatmap, (w, h))

        heatmap_uint8 = np.uint8(255 * resized_heatmap)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        overlay = (alpha * colormap_rgb + (1 - alpha) * orig_np).astype(np.uint8)
        return Image.fromarray(overlay)
