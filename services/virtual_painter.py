import re
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union, List

from models.wall_segmenter import WallSegmenter
from utils.image_utils import resize_image, save_result_image


PRESET_COLORS = {
    "浅蓝": (230, 200, 173),
    "天蓝": (250, 216, 135),
    "深蓝": (180, 100, 30),
    "薄荷绿": (210, 224, 180),
    "墨绿": (110, 110, 40),
    "浅粉": (225, 192, 203),
    "珊瑚红": (160, 128, 240),
    "酒红": (60, 40, 128),
    "米白": (220, 235, 245),
    "暖白": (200, 230, 250),
    "灰色": (160, 160, 160),
    "深灰": (90, 90, 90),
    "奶咖": (150, 170, 190),
    "燕麦": (180, 200, 210),
    "藕粉": (190, 180, 220),
    "鹅黄": (160, 230, 250),
    "紫色": (180, 130, 150),
    "橙色": (80, 150, 240)
}


def parse_color(color_input: str) -> Optional[Tuple[int, int, int]]:
    if not color_input:
        return None

    c = color_input.strip()

    if c in PRESET_COLORS:
        return PRESET_COLORS[c]

    cn_match = re.match(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$", c)
    if cn_match:
        h = cn_match.group(1)
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return (b, g, r)

    rgb_match = re.match(
        r"^[rgbRGB\(\)\s,]*(\d{1,3})[,\s]+(\d{1,3})[,\s]+(\d{1,3})", c
    )
    if rgb_match:
        r = int(rgb_match.group(1))
        g = int(rgb_match.group(2))
        b = int(rgb_match.group(3))
        if all(0 <= x <= 255 for x in (r, g, b)):
            return (b, g, r)

    for name, bgr in PRESET_COLORS.items():
        if c in name or name in c:
            return bgr

    return None


def color_to_hex(bgr: Tuple[int, int, int]) -> str:
    b, g, r = bgr
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


class VirtualPainter:
    def __init__(self):
        self.segmenter = WallSegmenter()

    def process(self,
                 image: np.ndarray,
                 target_color: Union[str, Tuple[int, int, int]],
                 blend_strength: float = 0.85,
                 preserve_shading: bool = True,
                 preserve_texture: bool = True,
                 color_bleed: int = 2) -> Dict[str, Any]:
        parsed_color = target_color if isinstance(target_color, tuple) else parse_color(target_color)
        if parsed_color is None:
            return {
                "success": False,
                "error": f"无法解析颜色: {target_color}，请使用HEX/RGB/预设色名",
                "available_presets": list(PRESET_COLORS.keys())
            }

        working = resize_image(image, max_size=(1600, 1600))
        h, w = working.shape[:2]

        seg_result = self.segmenter.segment(working)
        mask = seg_result["mask"]

        if np.mean(mask > 0) < 0.005:
            seg_result2 = self.segmenter.segment(
                cv2.convertScaleAbs(working, alpha=1.3, beta=20)
            )
            if np.mean(seg_result2["mask"] > 0) > np.mean(mask > 0):
                mask = seg_result2["mask"]
                seg_result = seg_result2

        if np.mean(mask > 0) < 0.005:
            return {
                "success": False,
                "error": "未能识别到墙面区域，请上传包含清晰墙面的室内照片",
                "available_presets": list(PRESET_COLORS.keys())
            }

        if color_bleed > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (color_bleed * 2 + 1, color_bleed * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)
        mask_float = (mask > 0).astype(np.float32)
        mask_smooth = cv2.GaussianBlur(
            mask_float, (color_bleed * 6 + 1 if color_bleed else 5,
                        color_bleed * 6 + 1 if color_bleed else 5), 0
        )[:, :, np.newaxis]

        painted, shading_map, texture_map = self._recolor_wall(
            working, mask_smooth, parsed_color,
            blend_strength, preserve_shading, preserve_texture
        )

        before_crop = self._visualize_before_after(working, mask)
        after_crop = self._visualize_before_after(painted, mask)

        result_filename = save_result_image(painted, prefix="repaint")
        before_filename = save_result_image(before_crop, prefix="repaint_before")
        mask_filename = save_result_image(
            self.segmenter.visualize_mask(working, seg_result["mask"]),
            prefix="repaint_mask"
        )

        return {
            "success": True,
            "target_color_bgr": parsed_color,
            "target_color_hex": color_to_hex(parsed_color),
            "segmentation": {
                "method": seg_result["method"],
                "confidence": seg_result["confidence"],
                "wall_area_ratio": seg_result["wall_area_ratio"]
            },
            "result_image_url": f"/static/{result_filename}",
            "before_url": f"/static/{before_filename}",
            "mask_overlay_url": f"/static/{mask_filename}",
            "shading_map_stats": {
                "min_shading": float(np.min(shading_map)),
                "max_shading": float(np.max(shading_map)),
                "mean_shading": float(np.mean(shading_map))
            },
            "texture_preserved": preserve_texture,
            "image_size": {"width": w, "height": h}
        }

    def _recolor_wall(self,
                       image: np.ndarray,
                       mask_smooth: np.ndarray,
                       target_bgr: Tuple[int, int, int],
                       blend_strength: float,
                       preserve_shading: bool,
                       preserve_texture: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        img_float = image.astype(np.float32) / 255.0
        target = np.array(target_bgr, dtype=np.float32) / 255.0

        shading_map = np.ones((h, w), dtype=np.float32)
        texture_map = np.ones((h, w), dtype=np.float32)

        if preserve_shading or preserve_texture:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            if preserve_shading:
                large_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=min(h, w) * 0.04)
                mean_val = np.mean(large_blur) + 1e-6
                shading_map = (large_blur / mean_val).astype(np.float32)
                shading_map = np.clip(shading_map, 0.4, 1.5)

            if preserve_texture:
                small_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
                texture_detail = gray / (small_blur + 1e-6)
                texture_map = np.clip(texture_detail, 0.85, 1.15).astype(np.float32)

        combined_modulation = (shading_map if preserve_shading else 1.0) * \
                              (texture_map if preserve_texture else 1.0)
        modulation_3ch = combined_modulation[:, :, np.newaxis]

        flat_color = np.ones_like(img_float) * target
        lit_color = np.clip(flat_color * modulation_3ch, 0.0, 1.0)

        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_tgt = np.zeros_like(hsv_img)
        hsv_tgt[:, :, 0] = cv2.cvtColor(
            (lit_color * 255).astype(np.uint8), cv2.COLOR_BGR2HSV
        )[:, :, 0].astype(np.float32)
        hsv_tgt[:, :, 1] = hsv_img[:, :, 1] * 0.3 + \
                           cv2.cvtColor((lit_color * 255).astype(np.uint8),
                                        cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) * 0.7
        hsv_tgt[:, :, 2] = hsv_img[:, :, 2] * 0.4 + \
                           (cv2.cvtColor((lit_color * 255).astype(np.uint8),
                                          cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32) *
                            combined_modulation) * 0.6
        hsv_blend = cv2.cvtColor(np.clip(hsv_tgt, 0, 255).astype(np.uint8),
                                  cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

        alpha = blend_strength * mask_smooth
        result = img_float * (1.0 - alpha) + hsv_blend * alpha
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

        return result, shading_map, texture_map

    def _visualize_before_after(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return image.copy()

    def batch_process(self,
                       image: np.ndarray,
                       color_list: List[Union[str, Tuple[int, int, int]]],
                       **kwargs) -> Dict[str, Any]:
        results = []
        for color in color_list:
            result = self.process(image, color, **kwargs)
            results.append({
                "color": color if isinstance(color, str) else color_to_hex(color),
                **result
            })
        return {
            "count": len(results),
            "results": results
        }
