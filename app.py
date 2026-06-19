import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    UPLOAD_DIR, STATIC_DIR,
    MAX_CONCURRENT_REQUESTS, REQUEST_QUEUE_TIMEOUT, ENABLE_AUTO_ENHANCE
)
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
    get_preprocess_info,
    cleanup_old_files
)

inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
pending_requests = 0

style_classifier: Optional[StyleClassifier] = None
furniture_detector: Optional[FurnitureDetector] = None
recommender: Optional[FurnitureRecommender] = None


def get_style_classifier() -> StyleClassifier:
    global style_classifier
    if style_classifier is None:
        style_classifier = StyleClassifier()
    return style_classifier


def get_furniture_detector() -> FurnitureDetector:
    global furniture_detector
    if furniture_detector is None:
        furniture_detector = FurnitureDetector()
    return furniture_detector


def get_recommender() -> FurnitureRecommender:
    global recommender
    if recommender is None:
        recommender = FurnitureRecommender()
    return recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    global style_classifier, furniture_detector, recommender
    print("[Server] 正在初始化 AI 模型（常驻内存模式）...")
    style_classifier = get_style_classifier()
    furniture_detector = get_furniture_detector()
    recommender = get_recommender()
    print(f"[Server] 最大并发推理数: {MAX_CONCURRENT_REQUESTS}")
    print(f"[Server] 显存/内存优化: 模型单例复用 + Semaphore 限流")
    cleanup_old_files(UPLOAD_DIR, max_age_hours=24)
    cleanup_old_files(STATIC_DIR, max_age_hours=24)
    print("[Server] 启动完成，服务已就绪")
    yield
    print("[Server] 正在关闭服务，释放资源...")
    style_classifier = None
    furniture_detector = None
    recommender = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[Server] 资源已释放")


app = FastAPI(
    title="室内装修风格识别与家具推荐 AI 服务",
    description="上传室内照片，自动识别装修风格、检测家具位置、推荐搭配家具。"
                "内置低光照图像增强（Retinex + CLAHE + 白平衡），高并发模型常驻 + 信号量限流。",
    version="1.1.0",
    lifespan=lifespan
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


class StyleInfoResponse(BaseModel):
    style_name: str
    description: str
    color_palette: list
    materials: list
    recommended_furniture: dict
    avoid: list


async def acquire_inference_lock():
    global pending_requests
    pending_requests += 1
    try:
        await asyncio.wait_for(
            inference_semaphore.acquire(),
            timeout=REQUEST_QUEUE_TIMEOUT
        )
    except asyncio.TimeoutError:
        pending_requests -= 1
        raise HTTPException(
            status_code=503,
            detail=f"服务器繁忙，当前等待请求过多（{pending_requests}），请稍后重试"
        )


def release_inference_lock():
    global pending_requests
    inference_semaphore.release()
    pending_requests = max(0, pending_requests - 1)


@app.middleware("http")
async def memory_pressure_monitor(request, call_next):
    import gc
    if request.url.path in ("/", "/health", "/styles", "/docs", "/openapi.json", "/redoc"):
        return await call_next(request)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    response = await call_next(request)
    return response


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "models_loaded": {
            "style_classifier": style_classifier is not None,
            "furniture_detector": furniture_detector is not None,
            "recommender": recommender is not None
        },
        "concurrency": {
            "max_concurrent": MAX_CONCURRENT_REQUESTS,
            "current_pending": pending_requests,
            "available_slots": MAX_CONCURRENT_REQUESTS - pending_requests + (
                MAX_CONCURRENT_REQUESTS - inference_semaphore._value
            )
        }
    }


@app.get("/")
async def root():
    return {
        "service": "室内装修风格识别与家具推荐 AI 服务",
        "version": "1.1.0",
        "features": [
            "ResNet50 装修风格分类（8种风格）",
            "YOLOv8 家具目标检测与定位",
            "预置搭配规则库 + 推荐引擎",
            "低光照/逆光图像增强（Retinex + CLAHE + 白平衡）",
            "模型常驻内存 + 信号量并发限流"
        ],
        "endpoints": {
            "GET /health": "服务健康状态与并发信息",
            "POST /analyze": "完整分析（风格识别+家具检测+搭配推荐）",
            "POST /classify-style": "仅识别装修风格",
            "POST /detect-furniture": "仅检测家具",
            "GET /styles": "所有支持的装修风格列表",
            "GET /styles/{style_name}": "指定风格的详细搭配规则"
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
    enable_enhance: bool = ENABLE_AUTO_ENHANCE,
    classifier: StyleClassifier = Depends(get_style_classifier)
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式，请上传 jpg/jpeg/png/bmp 图片")

    file_data = await file.read()
    valid, msg, image = validate_image(file_data)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    original_info = get_preprocess_info(image)

    try:
        await acquire_inference_lock()
        start_time = time.time()
        processed_image = preprocess_for_inference(image, enhance=enable_enhance)
        style_result = classifier.predict(processed_image)
        inference_time = round(time.time() - start_time, 3)
    finally:
        release_inference_lock()

    enhanced_info = get_preprocess_info(processed_image) if enable_enhance else original_info

    return {
        "filename": file.filename,
        "image_info": get_image_info(image),
        "preprocessing": {
            "original_lighting": original_info,
            "enhanced_lighting": enhanced_info,
            "enhance_enabled": enable_enhance
        },
        "style_result": style_result,
        "inference_time_seconds": inference_time
    }


@app.post("/detect-furniture")
async def detect_furniture(
    file: UploadFile = File(...),
    draw_boxes: bool = True,
    enable_enhance: bool = ENABLE_AUTO_ENHANCE,
    detector: FurnitureDetector = Depends(get_furniture_detector)
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式，请上传 jpg/jpeg/png/bmp 图片")

    file_data = await file.read()
    valid, msg, image = validate_image(file_data)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    original_info = get_preprocess_info(image)

    try:
        await acquire_inference_lock()
        start_time = time.time()
        processed_image = preprocess_for_inference(image, enhance=enable_enhance)
        detection_result = detector.detect(processed_image)
        inference_time = round(time.time() - start_time, 3)
    finally:
        release_inference_lock()

    result_image_url = None
    if draw_boxes and detection_result["count"] > 0:
        annotated = detector.draw_detections(processed_image, detection_result["detections"])
        result_filename = save_result_image(annotated, prefix="detection")
        result_image_url = f"/static/{result_filename}"

    enhanced_info = get_preprocess_info(processed_image) if enable_enhance else original_info

    return {
        "filename": file.filename,
        "image_info": get_image_info(image),
        "preprocessing": {
            "original_lighting": original_info,
            "enhanced_lighting": enhanced_info,
            "enhance_enabled": enable_enhance
        },
        "detection_result": detection_result,
        "annotated_image_url": result_image_url,
        "inference_time_seconds": inference_time
    }


@app.post("/analyze")
async def full_analysis(
    file: UploadFile = File(...),
    draw_boxes: bool = True,
    enable_enhance: bool = ENABLE_AUTO_ENHANCE,
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

    original_info = get_preprocess_info(image)

    try:
        await acquire_inference_lock()
        total_start = time.time()
        processed_image = preprocess_for_inference(image, enhance=enable_enhance)

        t1 = time.time()
        style_result = classifier.predict(processed_image)
        style_time = round(time.time() - t1, 3)

        t2 = time.time()
        detection_result = detector.detect(processed_image)
        detection_time = round(time.time() - t2, 3)

        t3 = time.time()
        recommendations = recommender.generate_recommendations(style_result, detection_result)
        rec_time = round(time.time() - t3, 3)
    finally:
        release_inference_lock()

    result_image_url = None
    if draw_boxes and detection_result["count"] > 0:
        annotated = detector.draw_detections(processed_image, detection_result["detections"])
        result_filename = save_result_image(annotated, prefix="analysis")
        result_image_url = f"/static/{result_filename}"

    saved_path, saved_name = save_uploaded_file(file_data, file.filename)

    total_time = round(time.time() - total_start, 3)
    enhanced_info = get_preprocess_info(processed_image) if enable_enhance else original_info

    return {
        "filename": file.filename,
        "saved_filename": saved_name,
        "image_info": get_image_info(image),
        "preprocessing": {
            "original_lighting": original_info,
            "enhanced_lighting": enhanced_info,
            "enhance_enabled": enable_enhance
        },
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


@app.post("/preprocess-preview")
async def preprocess_preview(
    file: UploadFile = File(...)
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式，请上传 jpg/jpeg/png/bmp 图片")

    file_data = await file.read()
    valid, msg, image = validate_image(file_data)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    original_info = get_preprocess_info(image)
    resized = preprocess_for_inference(image, enhance=False)
    enhanced = preprocess_for_inference(image, enhance=True)
    enhanced_info = get_preprocess_info(enhanced)

    orig_filename = save_result_image(resized, prefix="original")
    enhanced_filename = save_result_image(enhanced, prefix="enhanced")

    return {
        "filename": file.filename,
        "original_image_url": f"/static/{orig_filename}",
        "enhanced_image_url": f"/static/{enhanced_filename}",
        "lighting_before": original_info,
        "lighting_after": enhanced_info
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
