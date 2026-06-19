import os
import sys
import requests
import argparse


BASE_URL = "http://127.0.0.1:8000"


def test_root():
    print("=== 测试 GET / ===")
    resp = requests.get(f"{BASE_URL}/")
    print(f"状态码: {resp.status_code}")
    print(resp.json())
    print()


def test_list_styles():
    print("=== 测试 GET /styles ===")
    resp = requests.get(f"{BASE_URL}/styles")
    print(f"状态码: {resp.status_code}")
    data = resp.json()
    print(f"共 {data['count']} 种风格:")
    for s in data["styles"]:
        print(f"  - {s['name']}")
    print()


def test_style_detail(style_name="北欧风格"):
    print(f"=== 测试 GET /styles/{style_name} ===")
    resp = requests.get(f"{BASE_URL}/styles/{style_name}")
    print(f"状态码: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"风格: {data['style_name']}")
        print(f"描述: {data['description']}")
        print(f"推荐色系: {data['color_palette']}")
        print(f"推荐材质: {data['materials']}")
    print()


def test_classify_style(image_path):
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}，跳过风格分类测试")
        return
    print("=== 测试 POST /classify-style ===")
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        resp = requests.post(f"{BASE_URL}/classify-style", files=files)
    print(f"状态码: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"图片尺寸: {data['image_info']['width']}x{data['image_info']['height']}")
        print(f"主风格: {data['style_result']['primary_style']}")
        print(f"置信度: {data['style_result']['primary_confidence']}")
        print(f"Top-3: {data['style_result']['top_k']}")
        print(f"推理耗时: {data['inference_time_seconds']}s")
    print()


def test_detect_furniture(image_path):
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}，跳过家具检测测试")
        return
    print("=== 测试 POST /detect-furniture ===")
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        resp = requests.post(f"{BASE_URL}/detect-furniture", files=files, data={"draw_boxes": "true"})
    print(f"状态码: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        det = data["detection_result"]
        print(f"检测到 {det['count']} 个家具:")
        for item in det["detections"]:
            print(f"  - {item['label']} (置信度: {item['confidence']})")
        if data.get("annotated_image_url"):
            print(f"标注图片: {BASE_URL}{data['annotated_image_url']}")
    print()


def test_analyze(image_path):
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}，跳过完整分析测试")
        return
    print("=== 测试 POST /analyze ===")
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        resp = requests.post(f"{BASE_URL}/analyze", files=files, data={"draw_boxes": "true"})
    print(f"状态码: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()

        print(f"\n--- 风格识别 ---")
        sr = data["style_recognition"]["result"]
        print(f"主风格: {sr['primary_style']} (置信度: {sr['primary_confidence']})")

        print(f"\n--- 家具检测 ---")
        fd = data["furniture_detection"]["result"]
        print(f"共检测到 {fd['count']} 件家具:")
        for item in fd["detections"]:
            print(f"  - {item['label']}")

        print(f"\n--- 搭配推荐 ---")
        rec = data["furniture_recommendations"]
        print(f"风格描述: {rec['style_analysis']['style_description']}")
        print(f"推荐色系: {rec['style_analysis']['color_palette']}")
        print(f"推荐材质: {rec['style_analysis']['recommended_materials']}")
        print(f"\n缺少的家具: {rec['missing_essential_furniture']}")
        print(f"\n具体推荐:")
        for r in rec["specific_recommendations"][:5]:
            print(f"  [{r['priority']}] {r['furniture_type']}: {r['suggested_style']}")
            print(f"      颜色: {r['suggested_colors']}, 材质: {r['suggested_materials']}")

        print(f"\n总耗时: {data['total_processing_time_seconds']}s")
        if data["furniture_detection"].get("annotated_image_url"):
            print(f"标注图片: {BASE_URL}{data['furniture_detection']['annotated_image_url']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="AI 服务客户端测试")
    parser.add_argument("--image", type=str, default="", help="测试图片路径")
    parser.add_argument("--tests", type=str, default="all",
                        choices=["all", "root", "styles", "classify", "detect", "analyze"],
                        help="要运行的测试")
    args = parser.parse_args()

    try:
        if args.tests in ["all", "root"]:
            test_root()
        if args.tests in ["all", "styles"]:
            test_list_styles()
            test_style_detail()
        if args.tests in ["all", "classify"]:
            test_classify_style(args.image)
        if args.tests in ["all", "detect"]:
            test_detect_furniture(args.image)
        if args.tests in ["all", "analyze"]:
            test_analyze(args.image)
    except requests.ConnectionError:
        print("❌ 无法连接到服务器，请先启动服务: python app.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
