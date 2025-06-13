import cv2
import time
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Img
import sys
import time

# 固定到onnx路径·
# sys.path.append('./paddle_to_onnx/onnx')

model = ONNXPaddleOcr(use_angle_cls=False, use_gpu=True)

for i in range(1,10):
    img = cv2.imread("./1.png")
    s = time.time()
    result = model.ocr(img)
    e = time.time()
    print("total time: {:.3f}".format(e - s))
# print("result:", result)
# for box in result[0]:
    # print(box)

# sav2Img(img, result, name=str(time.time()) + ".jpg")
