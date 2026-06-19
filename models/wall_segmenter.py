import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy import ndimage

from config import DEVICE


ADE20K_WALL_CLASS_INDEX = 0


class WallSegmenter:
    def __init__(self):
        self.device = DEVICE
        self.dl_model = None
        self._try_load_deep_model()

    def _try_load_deep_model(self):
        try:
            from torchvision import models
            import torch
            self.dl_model = models.segmentation.deeplabv3_resnet50(
                pretrained=False,
                num_classes=150,
                aux_loss=False
            )
            weight_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models", "deeplabv3_ade20k.pth"
            )
            if os.path.exists(weight_path):
                state = torch.load(weight_path, map_location=self.device)
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                self.dl_model.load_state_dict(state, strict=False)
            self.dl_model = self.dl_model.to(self.device)
            self.dl_model.eval()
            print("[WallSegmenter] DeepLabV3 分割模型加载成功")
        except Exception as e:
            print(f"[WallSegmenter] 深度学习模型不可用，将使用传统CV方案: {e}")
            self.dl_model = None

    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]

        if self.dl_model is not None:
            mask, score = self._deep_segment(image)
            if score > 0.05:
                return {
                    "mask": mask,
                    "confidence": round(float(score), 4),
                    "method": "deep_lab_v3",
                    "wall_area_ratio": round(float(np.mean(mask > 0)), 4)
                }

        mask, score = self._traditional_segment(image)
        return {
            "mask": mask,
            "confidence": round(float(score), 4),
            "method": "traditional_cv",
            "wall_area_ratio": round(float(np.mean(mask > 0)), 4)
        }

    def _deep_segment(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        try:
            import torch
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.dl_model(tensor)["out"][0]
                probs = torch.softmax(out, dim=0)
                wall_prob = probs[ADE20K_WALL_CLASS_INDEX]
                mask_512 = (wall_prob > 0.3).cpu().numpy().astype(np.uint8) * 255
                score = float(wall_prob.mean().cpu().numpy())

            h, w = image.shape[:2]
            mask = cv2.resize(mask_512, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 127).astype(np.uint8) * 255

            mask = self._refine_mask(image, mask)
            return mask, score

        except Exception as e:
            print(f"[WallSegmenter] 深度学习分割失败: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8), 0.0

    def _traditional_segment(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 90)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        low_sat_mask = (hsv[:, :, 1] < 80).astype(np.uint8) * 255
        mid_val_mask = ((hsv[:, :, 2] > 60) & (hsv[:, :, 2] < 240)).astype(np.uint8) * 255
        smooth_mask = cv2.blur(gray, (15, 15))
        texture_var = cv2.blur(cv2.absdiff(gray, smooth_mask) ** 2, (15, 15))
        low_texture_mask = (texture_var < 300).astype(np.uint8) * 255

        wall_like = cv2.bitwise_and(low_sat_mask, mid_val_mask)
        wall_like = cv2.bitwise_and(wall_like, low_texture_mask)
        wall_like = cv2.bitwise_and(wall_like, cv2.bitwise_not(edges))

        top_crop = int(h * 0.05)
        left_crop = int(w * 0.02)
        seed_points = [
            (left_crop, top_crop),
            (w - left_crop, top_crop),
            (w // 2, top_crop),
            (w // 4, top_crop + 10),
            (3 * w // 4, top_crop + 10),
        ]

        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        for (sx, sy) in seed_points:
            sy = min(sy, h - 2)
            sx = min(sx, w - 2)
            if wall_like[sy, sx] > 0:
                cv2.floodFill(
                    wall_like.copy(), flood_mask, (sx, sy),
                    255, loDiff=(8, 20, 20), upDiff=(8, 20, 20),
                    flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
                )

        flood_mask_actual = flood_mask[1:-1, 1:-1]
        wall_like = cv2.bitwise_or(wall_like, flood_mask_actual * 255)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                 minLineLength=min(w, h) * 0.25, maxLineGap=20)
        wall_region_mask = np.zeros((h, w), dtype=np.uint8)
        if lines is not None:
            pts = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if angle < 10 or angle > 170:
                    pts.append((min(x1, x2), y1))
                    pts.append((max(x1, x2), y2))
            if pts:
                pts = np.array(pts)
                top_y = max(int(np.percentile([p[1] for p in pts], 10)), int(h * 0.05))
                wall_region_mask[top_y:h, :] = 255

        combined = cv2.bitwise_and(wall_like, wall_region_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            min_area = (h * w) * 0.03
            valid_masks = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    aspect = cw / max(ch, 1)
                    if aspect > 0.4 and y < h * 0.6:
                        tmp = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(tmp, [cnt], -1, 255, -1)
                        valid_masks.append(tmp)
            if valid_masks:
                mask = valid_masks[0]
                for m in valid_masks[1:]:
                    mask = cv2.bitwise_or(mask, m)

        mask = self._refine_mask(image, mask)
        coverage = np.mean(mask > 0)
        score = 0.5 + coverage * 0.4 if coverage > 0.02 else 0.0
        return mask, score

    def _refine_mask(self, image: np.ndarray, mask: np.ndarray,
                      iterations: int = 3) -> np.ndarray:
        h, w = image.shape[:2]
        if np.mean(mask > 0) < 0.005:
            return mask

        mask = (mask > 127).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        grabcut_mask = np.where(mask > 127, 3, 2).astype(np.uint8)
        sure_fg = cv2.erode(mask, kernel, iterations=iterations)
        grabcut_mask[sure_fg > 0] = 1
        sure_bg = cv2.dilate(mask, kernel, iterations=iterations + 2)
        grabcut_mask[sure_bg == 0] = 0

        try:
            cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 5,
                        cv2.GC_INIT_WITH_MASK)
            mask_refined = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
                255, 0
            ).astype(np.uint8)

            if np.mean(mask_refined > 0) > 0.01:
                mask = mask_refined
        except Exception:
            pass

        mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        mask = (mask > 127).astype(np.uint8) * 255
        return mask

    def visualize_mask(self, image: np.ndarray, mask: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:
        overlay = image.copy()
        color_layer = np.zeros_like(image)
        color_layer[:, :] = (0, 165, 255)
        mask_3ch = (mask > 0).astype(np.uint8)[:, :, np.newaxis]
        blended = cv2.addWeighted(
            overlay, 1 - alpha,
            color_layer * mask_3ch, alpha, 0
        )
        result = np.where(mask_3ch > 0, blended, overlay)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result.astype(np.uint8)
