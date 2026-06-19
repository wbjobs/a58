import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STYLE_CLASSES = [
    "现代简约",
    "北欧风格",
    "新中式",
    "工业风格",
    "日式风格",
    "美式风格",
    "欧式古典",
    "地中海风格"
]

FURNITURE_CLASSES = [
    "沙发",
    "茶几",
    "餐桌",
    "椅子",
    "床",
    "衣柜",
    "柜子",
    "书桌",
    "书架",
    "电视柜",
    "床头柜",
    "梳妆台"
]

DEVICE = "cpu"

STYLE_MODEL_WEIGHTS = os.path.join(BASE_DIR, "models", "style_classifier.pth")

YOLO_MODEL_WEIGHTS = os.path.join(BASE_DIR, "models", "yolov8n.pt")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}

MAX_IMAGE_SIZE = (640, 640)

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

MAX_CONCURRENT_REQUESTS = 4
REQUEST_QUEUE_TIMEOUT = 60.0
ENABLE_AUTO_ENHANCE = True
