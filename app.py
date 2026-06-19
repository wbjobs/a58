import os
import time
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import UPLOAD_DIR, STATIC_DIR
from models.style_classifier import StyleClassifier
from models.furniture_detector import FurnitureDetector
from services.recommender import FurnitureRecommender
from services.style_rules import get_all_styles, get_style_recommendations
from utils.image_utils import (
    allowed_file,
    validate_image,
    save_uploaded_file,
    preprocess_for_inference,
    save_result_image,
    get_image_info,
    cleanup_old_files
)

app = FastAPI(
    title="室内装修风格识别与家具推荐 AI 服务",
    description="上传室内照片，自动识别装修风格、检测家具位置、推荐搭配家具",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

style_classifier = None
furniture_detector = None
recommender = None


def get_style_classifier():
    global style_classifier
    if style_classifier is None:
        style_classifier = StyleClassifier()
    return style_classifier


def get_furniture_detector():
    global furniture_detector
    if furniture_detector is None:
        furniture_detector = FurnitureDetector()
    return furniture_detector


def get_recommender():
    global recommender
    if recommender is None:
        recommender = FurnitureRecommender()
    return recommender


class StyleInfoResponse(BaseModel):
    style_name: str
    description: str
    color_palette: list
    materials: list
    recommended_furniture: dict
    avoid: list


@app.on_event("startup")
async def startup_event():
    print("[Server] 正在初始化 AI 模型...")
    get_style_classifier()
    get_furniture_detector()
    get_recommender()
    cleanup_old_files(UPLOAD_DIR, max_age_hours=24)
    cleanup_old_files(STATIC_DIR, max_age_hours=24)
    print("[Server] AI 模型加载完成，服务已就绪")


@app.get("/")
async def root():
    return {
        "service": "室内装修风格识别与家具推荐 AI 服务",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "上传室内照片进行完整分析（风格识别+家具检测+搭配推荐）",
            "POST /classify-style": "仅识别装修风格",
            "POST /detect-furniture": "仅检测家具",
            "GET /styles": "获取所有支持的装修风格列表",
            "GET /styles/{style_name}": "获取指定风格的详细搭配规则"
        }
    }


@app.get("/styles")
async def list_styles():
    styles = get_all_styles()
    return {
        "count": len(styles),
        "styles": [
            {
                "name": s,
                "detail": f"/styles/{s}"
            }
            for s in styles
        ]
    }


@app.get("/styles/{style_name}")
async def get_style_detail(style_name: str):
    info = get_style_recommendations(style_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"未找到风格: {style_name}")
    return {
        "style_name": style_name,
        **info
    }


@app.post("/classify-style")
async def classify_style(
    file: UploadFile = File(...),
    classifier: StyleClassifier = Depends(get_style_classifier)
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式，请上传 jpg/jpeg/png/bmp 图片")

    file_data = await file.read()
    valid, msg, image = validate_image(file_data)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    start_time = time.time()
    processed_image = preprocess_for_inference(image)
    style_result = classifier.predict(processed_image)
    inference_time = round(time.time() - start_time, 3)

    return {
        "filename": file.filename,
        "image_info": get_image_info(image),
        "style_result": style_result,
        "inference_time_seconds": inference_time
    }


@app.post("/detect-furniture")
async def detect_furniture(
    file: UploadFile = File(...),
    draw_boxes: bool = True,
    detector: FurnitureDetector = Depends(get_furniture_detector)
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式，请上传 jpg/jpeg/png/bmp 图片")

    file_data = await file.read()
    valid, msg, image = validate_image(file_data)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    start_time = time.time()
    processed_image = preprocess_for_inference(image)
    detection_result = detector.detect(processed_image)
    inference_time = round(time.time() - start_time, 3)

    result_image_url = None
    if draw_boxes and detection_result["count"] > 0:
        annotated = detector.draw_detections(processed_image, detection_result["detections"])
        result_filename = save_result_image(annotated, prefix="detection")
        result_image_url = f"/static/{result_filename}"

    return {
        "filename": file.filename,
        "image_info": get_image_info(image),
        "detection_result": detection_result,
        "annotated_image_url": result_image_url,
        "inference_time_seconds": inference_time
    }


@app.post("/analyze")
async def full_analysis(
    file: UploadFile = File(...),
    draw_boxes: bool = True,
    classifier: StyleClassifier = Depends(get_style_classifier),
    detector: FurnitureDetector = Depends(get_furniture_detector),
    recommender: FurnitureRecommender = Depends(get_recommender)
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式，请上传 jpg/jpeg/png/bmp 图片")

    file_data = await file.read()
    valid, msg, image = validate_image(file_data)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    total_start = time.time()
    processed_image = preprocess_for_inference(image)

    t1 = time.time()
    style_result = classifier.predict(processed_image)
    style_time = round(time.time() - t1, 3)

    t2 = time.time()
    detection_result = detector.detect(processed_image)
    detection_time = round(time.time() - t2, 3)

    t3 = time.time()
    recommendations = recommender.generate_recommendations(style_result, detection_result)
    rec_time = round(time.time() - t3, 3)

    result_image_url = None
    if draw_boxes and detection_result["count"] > 0:
        annotated = detector.draw_detections(processed_image, detection_result["detections"])
        result_filename = save_result_image(annotated, prefix="analysis")
        result_image_url = f"/static/{result_filename}"

    saved_path, saved_name = save_uploaded_file(file_data, file.filename)

    total_time = round(time.time() - total_start, 3)

    return {
        "filename": file.filename,
        "saved_filename": saved_name,
        "image_info": get_image_info(image),
        "style_recognition": {
            "result": style_result,
            "processing_time_seconds": style_time
        },
        "furniture_detection": {
            "result": detection_result,
            "annotated_image_url": result_image_url,
            "processing_time_seconds": detection_time
        },
        "furniture_recommendations": recommendations,
        "recommendation_processing_time_seconds": rec_time,
        "total_processing_time_seconds": total_time
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
