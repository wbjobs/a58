import os
import uuid
import cv2
import numpy as np
from typing import Tuple, Optional

from config import MAX_IMAGE_SIZE, ALLOWED_EXTENSIONS, UPLOAD_DIR, STATIC_DIR


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def save_uploaded_file(file_data: bytes, filename: str) -> Tuple[str, str]:
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    ext = filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(save_path, "wb") as f:
        f.write(file_data)

    return save_path, unique_name


def read_image(image_path: str) -> Optional[np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        return None
    return image


def resize_image(image: np.ndarray, max_size: Tuple[int, int] = MAX_IMAGE_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    max_w, max_h = max_size

    if w <= max_w and h <= max_h:
        return image

    scale = min(max_w / w, max_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_for_inference(image: np.ndarray) -> np.ndarray:
    processed = resize_image(image)
    return processed


def save_result_image(image: np.ndarray, prefix: str = "result") -> str:
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR, exist_ok=True)

    filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(STATIC_DIR, filename)
    cv2.imwrite(save_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return filename


def get_image_info(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1

    return {
        "width": w,
        "height": h,
        "channels": channels,
        "aspect_ratio": round(w / h, 3) if h > 0 else 0
    }


def bytes_to_numpy(file_data: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def validate_image(file_data: bytes) -> Tuple[bool, str, Optional[np.ndarray]]:
    try:
        image = bytes_to_numpy(file_data)
        if image is None:
            return False, "无法解码图片，请检查文件格式", None

        h, w = image.shape[:2]
        if h < 50 or w < 50:
            return False, f"图片尺寸过小 ({w}x{h})，至少需要 50x50", None

        if h > 8000 or w > 8000:
            return False, f"图片尺寸过大 ({w}x{h})，最大支持 8000x8000", None

        return True, "图片有效", image

    except Exception as e:
        return False, f"图片解析失败: {str(e)}", None


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    import time
    if not os.path.exists(directory):
        return

    now = time.time()
    cutoff = now - (max_age_hours * 3600)

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_mtime = os.path.getmtime(filepath)
            if file_mtime < cutoff:
                try:
                    os.remove(filepath)
                except OSError:
                    pass


def estimate_brightness(image: np.ndarray) -> float:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(np.mean(gray))


def is_low_light(image: np.ndarray, threshold: float = 80.0) -> bool:
    brightness = estimate_brightness(image)
    return brightness < threshold


def is_overexposed(image: np.ndarray, threshold: float = 230.0) -> bool:
    brightness = estimate_brightness(image)
    return brightness > threshold


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(image: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    if gamma is None:
        brightness = estimate_brightness(image)
        if brightness < 50:
            gamma = 2.2
        elif brightness < 80:
            gamma = 1.7
        elif brightness < 110:
            gamma = 1.3
        elif brightness > 220:
            gamma = 0.6
        elif brightness > 190:
            gamma = 0.8
        else:
            gamma = 1.0

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def adjust_brightness_contrast(image: np.ndarray,
                                brightness: Optional[int] = None,
                                contrast: Optional[int] = None) -> np.ndarray:
    if brightness is None or contrast is None:
        current_brightness = estimate_brightness(image)
        target_brightness = 140.0
        brightness = int(target_brightness - current_brightness)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        current_std = float(np.std(gray))
        target_std = 65.0
        contrast = int((target_std / max(current_std, 1.0)) * 127 - 127)

    brightness = max(-255, min(255, brightness))
    contrast = max(-127, min(127, contrast))

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255.0
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return np.clip(image, 0, 255).astype(np.uint8)


def correct_white_balance(image: np.ndarray) -> np.ndarray:
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = result.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def apply_retinex(image: np.ndarray, sigma_list: Tuple[int, ...] = (15, 80, 250)) -> np.ndarray:
    def single_scale_retinex(img, sigma):
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex = cv2.log(img + 1.0) - cv2.log(blur + 1.0)
        return retinex

    img_float = image.astype(np.float64) + 1.0
    retinex = np.zeros_like(img_float)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img_float, sigma)
    retinex = retinex / len(sigma_list)

    for i in range(3):
        channel = retinex[:, :, i]
        channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel) + 1e-8)
        retinex[:, :, i] = channel * 255.0

    return retinex.astype(np.uint8)


def denoise_image(image: np.ndarray) -> np.ndarray:
    h = 5 if is_low_light(image) else 3
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)


def enhance_low_light(image: np.ndarray, aggressive: bool = True) -> np.ndarray:
    brightness = estimate_brightness(image)
    enhanced = image.copy()

    if aggressive and brightness < 100:
        enhanced = apply_retinex(enhanced)
    elif brightness < 50:
        enhanced = apply_retinex(enhanced, sigma_list=(30, 100, 300))

    enhanced = correct_white_balance(enhanced)
    enhanced = apply_gamma_correction(enhanced)
    enhanced = adjust_brightness_contrast(enhanced)
    enhanced = apply_clahe(enhanced, clip_limit=2.5, grid_size=8)
    enhanced = denoise_image(enhanced)

    return enhanced


def preprocess_for_inference(image: np.ndarray,
                              enhance: bool = True) -> np.ndarray:
    processed = resize_image(image)

    if enhance:
        brightness = estimate_brightness(processed)
        if brightness < 100 or brightness > 220:
            processed = enhance_low_light(processed, aggressive=(brightness < 70))

    return processed


def get_preprocess_info(image: np.ndarray) -> dict:
    brightness = estimate_brightness(image)
    low_light = is_low_light(image)
    overexposed = is_overexposed(image)

    label = "正常光照"
    if low_light:
        if brightness < 40:
            label = "严重暗光"
        elif brightness < 60:
            label = "中等暗光"
        else:
            label = "轻度暗光"
    elif overexposed:
        label = "过曝"

    return {
        "brightness": round(brightness, 2),
        "lighting_condition": label,
        "is_low_light": low_light,
        "is_overexposed": overexposed,
        "enhancement_applied": low_light or overexposed
    }
