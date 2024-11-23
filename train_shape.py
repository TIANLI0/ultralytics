from ultralytics import YOLO
import json
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from TrainSync import sync_to_server


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


modelA = "../runs/detect/train10/weights/last.pt"
description = "分组训练实验3"
version = "Group 0.2-fix1"
dataset = "D:/develop/YoloDB/tt100k_yolo_Shape/tt100k.yaml"
deviceA = 0

model = YOLO(modelA)

model.add_callback("on_train_epoch_end", on_train_epoch_end)

results = model.train(
    data=dataset,
    epochs=30,
    workers=0,
    device=deviceA,
    imgsz=640,
    cache=True,
    plots=True,
)
