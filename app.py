import os
import gc
import cv2
import time
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import (
    UPLOAD_DIR, STATIC_DIR,
    MAX_CONCURRENT_REQUESTS, REQUEST_QUEUE_TIMEOUT, ENABLE_AUTO_ENHANCE,
    PAINT_BLEND_STRENGTH, PAINT_PRESERVE_SHADING, PAINT_PRESERVE_TEXTURE
)
from models.style_classifier import StyleClassifier
from models.furniture_detector import FurnitureDetector
from models.wall_segmenter import WallSegmenter
from services.recommender import FurnitureRecommender
from services.style_rules import get_all_styles, get_style_recommendations
from services.virtual_painter import VirtualPainter, PRESET_COLORS, parse_color, color_to_hex
from services.task_manager import get_task_manager, TaskStatus
from utils.image_utils import (
    allowed_file,
    validate_image,
    save_uploaded_file,
    preprocess_for_inference,
    save_result_image,
    get_image_info,
    get_preprocess_info,
    cleanup_old_files,
    bytes_to_numpy
)

inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
pending_requests = 0

style_classifier: Optional[StyleClassifier] = None
furniture_detector: Optional[FurnitureDetector] = None
recommender: Optional[FurnitureRecommender] = None
wall_segmenter: Optional[WallSegmenter] = None
virtual_painter: Optional[VirtualPainter] = None


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


def get_wall_segmenter() -> WallSegmenter:
    global wall_segmenter
    if wall_segmenter is None:
        wall_segmenter = WallSegmenter()
    return wall_segmenter


def get_virtual_painter() -> VirtualPainter:
    global virtual_painter
    if virtual_painter is None:
        virtual_painter = VirtualPainter()
    return virtual_painter


@asynccontextmanager
async def lifespan(app: FastAPI):
    global style_classifier, furniture_detector, recommender
    global wall_segmenter, virtual_painter
    print("[Server] 正在初始化 AI 模型（常驻内存模式）...")
    style_classifier = get_style_classifier()
    furniture_detector = get_furniture_detector()
    recommender = get_recommender()
    wall_segmenter = get_wall_segmenter()
    virtual_painter = get_virtual_painter()

    tm = get_task_manager()
    tm.register_handler("wall_repaint", _wall_repaint_task_handler)
    tm.register_handler("batch_wall_repaint", _batch_wall_repaint_task_handler)
    tm.start()

    print(f"[Server] 最大并发推理数: {MAX_CONCURRENT_REQUESTS}")
    print(f"[Server] 异步任务 Workers: {tm.max_workers}, Queue容量: {tm.max_queue}")
    print(f"[Server] 显存/内存优化: 模型单例复用 + Semaphore 限流")
    cleanup_old_files(UPLOAD_DIR, max_age_hours=24)
    cleanup_old_files(STATIC_DIR, max_age_hours=24)
    print("[Server] 启动完成，服务已就绪")
    yield
    print("[Server] 正在关闭服务，释放资源...")
    get_task_manager().stop()
    style_classifier = None
    furniture_detector = None
    recommender = None
    wall_segmenter = None
    virtual_painter = None
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


def _wall_repaint_task_handler(params: dict, progress_fn) -> dict:
    progress_fn(5, "正在解码图片")
    image_bytes = params.get("image_bytes")
    if isinstance(image_bytes, (bytes, bytearray)):
        image = bytes_to_numpy(bytes(image_bytes))
    else:
        image_path = params.get("image_path")
        image = cv2.imread(image_path) if image_path else None

    if image is None:
        raise ValueError("无法解码图片数据")

    progress_fn(20, "正在分割墙面区域")
    painter = get_virtual_painter()

    kwargs = {
        "blend_strength": params.get("blend_strength", PAINT_BLEND_STRENGTH),
        "preserve_shading": params.get("preserve_shading", PAINT_PRESERVE_SHADING),
        "preserve_texture": params.get("preserve_texture", PAINT_PRESERVE_TEXTURE),
        "color_bleed": params.get("color_bleed", 2)
    }

    progress_fn(45, "正在应用虚拟换色")
    result = painter.process(image, params.get("target_color", "浅蓝"), **kwargs)

    if not result["success"]:
        return result

    progress_fn(95, "正在保存结果")
    if "image_bytes" in params:
        del params["image_bytes"]
    gc.collect()

    progress_fn(100, "处理完成")
    return result


def _batch_wall_repaint_task_handler(params: dict, progress_fn) -> dict:
    progress_fn(5, "正在解码图片")
    image_bytes = params.get("image_bytes")
    if isinstance(image_bytes, (bytes, bytearray)):
        image = bytes_to_numpy(bytes(image_bytes))
    else:
        image_path = params.get("image_path")
        image = cv2.imread(image_path) if image_path else None

    if image is None:
        raise ValueError("无法解码图片数据")

    painter = get_virtual_painter()
    colors = params.get("colors", ["浅蓝", "米白", "灰色", "薄荷绿"])
    batch_result = painter.batch_process(
        image, colors,
        blend_strength=params.get("blend_strength", PAINT_BLEND_STRENGTH),
        preserve_shading=params.get("preserve_shading", PAINT_PRESERVE_SHADING),
        preserve_texture=params.get("preserve_texture", PAINT_PRESERVE_TEXTURE)
    )

    if "image_bytes" in params:
        del params["image_bytes"]
    gc.collect()
    return batch_result


@app.get("/colors/presets")
async def list_color_presets():
    return {
        "count": len(PRESET_COLORS),
        "presets": [
            {"name": name, "hex": color_to_hex(bgr), "bgr": list(bgr)}
            for name, bgr in PRESET_COLORS.items()
        ],
        "usage": "用于 /paint-wall 和 /paint-wall/async 接口的 target_color 参数"
    }


@app.post("/segment-wall")
async def segment_wall(
    file: UploadFile = File(...),
    enable_enhance: bool = ENABLE_AUTO_ENHANCE,
    segmenter: WallSegmenter = Depends(get_wall_segmenter)
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式")

    file_data = await file.read()
    valid, msg, image = validate_image(file_data)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    try:
        await acquire_inference_lock()
        start = time.time()
        working = preprocess_for_inference(image, enhance=enable_enhance)
        seg_result = segmenter.segment(working)
        elapsed = round(time.time() - start, 3)
    finally:
        release_inference_lock()

    overlay = segmenter.visualize_mask(working, seg_result["mask"])
    filename = save_result_image(overlay, prefix="wall_seg")

    return {
        "filename": file.filename,
        "image_info": get_image_info(image),
        "segmentation": {
            "method": seg_result["method"],
            "confidence": seg_result["confidence"],
            "wall_area_ratio": seg_result["wall_area_ratio"],
            "processing_time_seconds": elapsed
        },
        "mask_overlay_url": f"/static/{filename}"
    }


@app.post("/paint-wall")
async def paint_wall_sync(
    file: UploadFile = File(...),
    target_color: str = Form("浅蓝"),
    blend_strength: float = Form(PAINT_BLEND_STRENGTH),
    preserve_shading: bool = Form(PAINT_PRESERVE_SHADING),
    preserve_texture: bool = Form(PAINT_PRESERVE_TEXTURE),
    color_bleed: int = Form(2),
    painter: VirtualPainter = Depends(get_virtual_painter)
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式")

    parsed = parse_color(target_color)
    if parsed is None:
        raise HTTPException(
            status_code=400,
            detail=f"无法解析颜色: {target_color}，请使用HEX/RGB或查看/colors/presets"
        )

    file_data = await file.read()
    valid, msg, image = validate_image(file_data)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    try:
        await acquire_inference_lock()
        start = time.time()
        result = painter.process(
            image, target_color,
            blend_strength=blend_strength,
            preserve_shading=preserve_shading,
            preserve_texture=preserve_texture,
            color_bleed=color_bleed
        )
        elapsed = round(time.time() - start, 3)
    finally:
        release_inference_lock()

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "处理失败"))

    result["processing_time_seconds"] = elapsed
    result["target_color"] = target_color
    return result


@app.post("/paint-wall/async")
async def paint_wall_async(
    file: UploadFile = File(...),
    target_color: str = Form("浅蓝"),
    blend_strength: float = Form(PAINT_BLEND_STRENGTH),
    preserve_shading: bool = Form(PAINT_PRESERVE_SHADING),
    preserve_texture: bool = Form(PAINT_PRESERVE_TEXTURE),
    color_bleed: int = Form(2),
    callback_url: Optional[str] = Form(None)
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式")

    parsed = parse_color(target_color)
    if parsed is None:
        raise HTTPException(
            status_code=400,
            detail=f"无法解析颜色: {target_color}，请使用HEX/RGB或查看/colors/presets"
        )

    file_data = await file.read()
    valid, msg, image = validate_image(file_data)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    saved_path, saved_name = save_uploaded_file(file_data, file.filename)

    params = {
        "image_path": saved_path,
        "target_color": target_color,
        "blend_strength": blend_strength,
        "preserve_shading": preserve_shading,
        "preserve_texture": preserve_texture,
        "color_bleed": color_bleed,
        "original_filename": file.filename,
    }

    submit_result = get_task_manager().submit(
        "wall_repaint", params, callback_url=callback_url
    )
    if not submit_result["success"]:
        raise HTTPException(status_code=503, detail=submit_result["error"])

    return submit_result


@app.post("/paint-wall/batch-async")
async def paint_wall_batch_async(
    file: UploadFile = File(...),
    colors: str = Form("浅蓝,米白,灰色,薄荷绿"),
    blend_strength: float = Form(PAINT_BLEND_STRENGTH),
    preserve_shading: bool = Form(PAINT_PRESERVE_SHADING),
    preserve_texture: bool = Form(PAINT_PRESERVE_TEXTURE),
    callback_url: Optional[str] = Form(None)
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式")

    color_list = [c.strip() for c in colors.split(",") if c.strip()]
    for c in color_list:
        if parse_color(c) is None:
            raise HTTPException(
                status_code=400,
                detail=f"无法解析颜色: {c}，请使用HEX/RGB或查看/colors/presets"
            )

    file_data = await file.read()
    valid, msg, image = validate_image(file_data)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    saved_path, saved_name = save_uploaded_file(file_data, file.filename)

    params = {
        "image_path": saved_path,
        "colors": color_list,
        "blend_strength": blend_strength,
        "preserve_shading": preserve_shading,
        "preserve_texture": preserve_texture,
        "original_filename": file.filename,
    }

    submit_result = get_task_manager().submit(
        "batch_wall_repaint", params, callback_url=callback_url
    )
    if not submit_result["success"]:
        raise HTTPException(status_code=503, detail=submit_result["error"])

    return submit_result


@app.get("/tasks")
async def list_all_tasks(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    tm = get_task_manager()
    status_filter = TaskStatus(status) if status else None
    tasks = tm.list_tasks(status_filter=status_filter, limit=limit)
    return {
        "count": len(tasks),
        "queue_size": tm._queue.qsize(),
        "tasks": tasks
    }


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    status = get_task_manager().get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return status


@app.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    result = get_task_manager().get_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return result


@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    result = get_task_manager().cancel(task_id)
    if not result["success"] and "不存在" in result.get("error", ""):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/static/{filename}/download")
async def download_static(filename: str):
    path = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
