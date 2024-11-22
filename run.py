import cv2
import os
from ultralytics import YOLO

# 加载模型
model = YOLO("../runs/detect/train9/weights/best.pt")

# 设置窗口大小
window_name = "YOLO Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1080, 1080)

image_dir = "D:/develop/YoloDB/tt100k_2021/test"
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)

    results = model(image)

    for result in results:
        annotated_image = result.plot()
        cv2.imshow(window_name, annotated_image)

        key = cv2.waitKey(0)

        if key == 27:  # 按下ESC键退出
            break

cv2.destroyAllWindows()
