from ultralytics import YOLO

# 读取模型
model = YOLO("yolo11n.yaml")  # 从 YAML 文件构建模型
model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # 从 YAML 文件构建模型并加载权重

# 训练数据集
results = model.train(data="D:/develop/YoloDB/tt100k_yolo/tt100k.yaml", epochs=5, workers=0,device=0)