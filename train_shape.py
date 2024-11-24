from ultralytics import YOLO
import json
from datetime import datetime
import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from TrainSync import sync_to_server
from TrainUpload import upload_files


def on_train_epoch_end(trainer):
    """获取当前的训练详情并同步到服务器"""
    data = {}
    for key in dir(trainer):
        value = getattr(trainer, key)
        try:
            json.dumps(value)
            data[key] = value
        except (TypeError, OverflowError):
            continue

    current_epoch = trainer.epoch + 1

    data.update(
        {
            "model": modelA,
            "train_device": str(deviceA),
            "dataset": dataset,
            "description": description,
            "version": version,
        }
    )

    sync_to_server(data, current_epoch)


def on_model_save(trainer):
    """在模型保存时更新arg.yaml文件并上传文件"""
    # 查找最新的训练文件夹
    base_path = "../runs/detect"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    latest_folder = max(
        [
            os.path.join(base_path, d)
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ],
        key=os.path.getmtime,
    )

    # 更新arg.yaml文件
    arg_file = os.path.join(latest_folder, "args.yaml")
    with open(arg_file, "r") as f:
        args = yaml.safe_load(f)

    args.update(
        {
            "version": version,
            "train_device": "3060 laptop 6GB",
            "dataset": "TT100K Tianli",
            "description": description,
        }
    )

    with open(arg_file, "w") as f:
        yaml.safe_dump(args, f)

    # 上传文件
    upload_files(latest_folder, os.path.basename(latest_folder))


modelA = "../runs/detect/train14/weights/last.pt"
description = "分组分类实验1"
version = "Classification 0.1"
dataset = "D:/develop/YoloDB/tt100k_yolo/tt100k.yaml"
deviceA = 0

model = YOLO(modelA)

model.add_callback("on_train_epoch_end", on_train_epoch_end)
model.add_callback("on_train_end", on_model_save)

results = model.train(
    data=dataset,
    epochs=30,
    workers=0,
    device=deviceA,
    imgsz=640,
    cache=True,
    plots=True,
)
