import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from config import STYLE_CLASSES, DEVICE, STYLE_MODEL_WEIGHTS


class StyleClassifier:
    def __init__(self):
        self.device = torch.device(DEVICE)
        self.class_names = STYLE_CLASSES
        self.model = self._build_model()
        self.transform = self._build_transform()
        self._load_weights()

    def _build_model(self):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(self.class_names))
        )
        model = model.to(self.device)
        model.eval()
        return model

    def _build_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_weights(self):
        import os
        if os.path.exists(STYLE_MODEL_WEIGHTS):
            try:
                checkpoint = torch.load(STYLE_MODEL_WEIGHTS, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"[StyleClassifier] 权重加载成功: {STYLE_MODEL_WEIGHTS}")
            except Exception as e:
                print(f"[StyleClassifier] 权重加载失败，使用随机初始化: {e}")
        else:
            print(f"[StyleClassifier] 未找到权重文件 {STYLE_MODEL_WEIGHTS}，使用随机初始化")

    def predict(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[:, :, ::-1])
        elif isinstance(image, str):
            image = Image.open(image).convert('RGB')

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            scores, indices = torch.topk(probabilities, k=3)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "style": self.class_names[idx.item()],
                "confidence": round(float(score.item()), 4)
            })

        return {
            "primary_style": results[0]["style"],
            "primary_confidence": results[0]["confidence"],
            "top_k": results
        }
