from ultralytics import YOLO

# 加载模型
model = YOLO("../runs/detect/train7/weights/best.pt")

# 识别图片
results = model(
    source=0,
    device="0",
    show=True,
    conf=0.5, # 针对car这个模型，0.6是最佳阈值
)
