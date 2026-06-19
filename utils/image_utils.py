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
