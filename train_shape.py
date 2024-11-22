from ultralytics import YOLO

# 读取模型
model = YOLO("yolo11n")  # 加载预训练模型（推荐用于训练）

# 训练数据集
results = model.train(
    data="D:/develop/YoloDB/tt100k_yolo_Shape/tt100k.yaml",
    epochs=60,  # 训练轮数
    workers=0,  # 数据加载器工作线程数，windows下设置为0
    device=0,  # GPU id
    imgsz=640,
    # time=3, # 限制训练时间（小时）
    cache=True,  # 缓存数据集
    plots=True,  # 绘制结果
)
