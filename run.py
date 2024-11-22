from flask import Flask, request, jsonify, send_from_directory
import cv2
from ultralytics import YOLO
import os
import uuid
import base64
import numpy as np
import json

app = Flask(__name__)


def get_model_path(model_path2):
    return "../runs/detect/" + model_path2 + "/weights/best.pt"


def send_file(file_path):
    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path))


@app.route("/")
def index():
    with open("../web/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return html


@app.route("/results")
def result():
    with open("../web/results.html", "r", encoding="utf-8") as f:
        html = f.read()
    return html


@app.route("/get_train_file", methods=["GET"])
def get_file_list():
    path = request.args.get("path")
    print("Request received for path:", path)
    if not path:
        return jsonify({"message": "路径不能为空"}), 400
    directory = os.path.abspath("../runs/detect")

    file_list = []
    path = os.path.join(directory, path)

    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.relpath(os.path.join(root, file), directory))

    return jsonify(file_list), 200


@app.route("/get_model_list", methods=["GET"])
def get_model_list():
    with open("../info.json", "r", encoding="utf-8") as f:
        data = f.read()
    data = json.loads(data)
    return jsonify(data), 200


# 返回静态文件
@app.route("/static")
def get_static_file():
    path = request.args.get("path")
    print("Request received for path:", path)
    if not path:
        return jsonify({"message": "路径不能为空"}), 400
    directory = os.path.abspath("../runs/detect")

    file_path = os.path.join(directory, path)
    print("Directory:", file_path)
    return send_file(file_path)


@app.route("/configure", methods=["POST"])
def configure_model():
    data = request.json
    model_path2 = data.get("model_path")
    model_path = get_model_path(model_path2)
    print(model_path)
    if not model_path or not os.path.exists(model_path):
        return jsonify({"message": "路径不存在", "model_path": model_path}), 400

    return (
        jsonify({"message": "模型配置成功", "model_path": model_path}),
        200,
    )


# 上传图片并使用yolo检测
@app.route("/detect_image", methods=["POST"])
def detect_image():
    model_path = get_model_path(request.form.get("model_path"))
    model = YOLO(model_path)

    if not model:
        return jsonify({"message": "模型未配置"}), 400

    file = request.files["image"]
    file_path = os.path.join("uploads", str(uuid.uuid4()) + ".jpg")
    file.save(file_path)

    results = model(file_path)

    # 解析检测结果
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append(
                {
                    "bbox": box.xyxy.tolist(),
                    "confidence": box.conf.tolist(),
                    "class": result.names[box.cls.tolist()[0]],
                }
            )

    with open(file_path, "rb") as f:
        image = f.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    _, buffer = cv2.imencode(".jpg", image)
    image = base64.b64encode(buffer).decode("utf-8")

    # 返回检测结果
    return jsonify({"detections": detections, "image": image}), 200


# 上传路径并使用yolo检测
@app.route("/detect_image_path", methods=["POST"])
def detect_image_path():
    model_path = get_model_path(request.json.get("model_path"))
    model = YOLO(model_path)

    if not model:
        return jsonify({"message": "模型未配置"}), 400

    data = request.json
    image_path = data.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return jsonify({"message": "路径不存在", "image_path": image_path}), 400

    results = model(image_path)

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append(
                {
                    "bbox": box.xyxy.tolist(),
                    "confidence": box.conf.tolist(),
                    "class": result.names[box.cls.tolist()[0]],
                }
            )

    with open(image_path, "rb") as f:
        image = f.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    _, buffer = cv2.imencode(".jpg", image)
    image = base64.b64encode(buffer).decode("utf-8")

    # 返回检测结果
    return jsonify({"detections": detections, "image": image}), 200


if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True, port=5090)
