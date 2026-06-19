import os
import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_MODEL_WEIGHTS,
    FURNITURE_CLASSES,
    DEVICE,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD
)


COCO_FURNITURE_MAP = {
    "couch": "沙发",
    "chair": "椅子",
    "bed": "床",
    "dining table": "餐桌",
    "desk": "书桌",
    "toilet": "马桶",
    "tv": "电视柜",
    "laptop": "笔记本",
    "microwave": "微波炉",
    "oven": "烤箱",
    "toaster": "烤面包机",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "书籍",
    "clock": "时钟",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "毛绒玩具",
    "hair drier": "吹风机",
    "toothbrush": "牙刷"
}


class FurnitureDetector:
    def __init__(self):
        self.device = DEVICE
        self.conf_threshold = CONFIDENCE_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        self.model = self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(YOLO_MODEL_WEIGHTS):
                model = YOLO(YOLO_MODEL_WEIGHTS)
            else:
                model = YOLO("yolov8n.pt")
                print(f"[FurnitureDetector] 本地权重未找到，使用 yolov8n.pt (将自动下载)")
            print(f"[FurnitureDetector] YOLO 模型加载成功")
            return model
        except Exception as e:
            print(f"[FurnitureDetector] 模型加载失败: {e}")
            return None

    def detect(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"无法读取图片: {image}")

        if self.model is None:
            return self._mock_detection(image)

        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        result = results[0]

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = result.names[class_id]

            furniture_name = COCO_FURNITURE_MAP.get(class_name, class_name)

            detections.append({
                "label": furniture_name,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": int(round(x1)),
                    "y1": int(round(y1)),
                    "x2": int(round(x2)),
                    "y2": int(round(y2))
                },
                "area": int(round((x2 - x1) * (y2 - y1)))
            })

        detections.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "count": len(detections),
            "detections": detections,
            "image_size": {"width": image.shape[1], "height": image.shape[0]}
        }

    def draw_detections(self, image, detections):
        if isinstance(image, str):
            image = cv2.imread(image)

        output = image.copy()
        colors = self._generate_colors()

        for det in detections:
            bbox = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            color_idx = hash(label) % len(colors)
            color = colors[color_idx]

            cv2.rectangle(
                output,
                (bbox["x1"], bbox["y1"]),
                (bbox["x2"], bbox["y2"]),
                color,
                2
            )

            text = f"{label} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            cv2.rectangle(
                output,
                (bbox["x1"], bbox["y1"] - text_h - 10),
                (bbox["x1"] + text_w + 10, bbox["y1"]),
                color,
                -1
            )
            cv2.putText(
                output, text,
                (bbox["x1"] + 5, bbox["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )

        return output

    def _generate_colors(self):
        return [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]

    def _mock_detection(self, image):
        print("[FurnitureDetector] 警告：模型未加载，返回空检测结果")
        return {
            "count": 0,
            "detections": [],
            "image_size": {"width": image.shape[1], "height": image.shape[0]}
        }
