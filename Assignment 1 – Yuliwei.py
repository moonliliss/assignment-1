#NAME:yuliwei
#ID:3160576
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import os

# 手动设置 Tesseract-OCR 的路径
pytesseract.pytesseract_cmd = r'E:\\Tesser-OCR\\Tesser-OCR\\tesseract.exe'


# 创建保存识别结果的文件夹
output_dir = 'output_text'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 实时保存识别出的文本
def save_recognized_text(text, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"识别出的文本已保存到：{output_filename}")


# 图像预处理：锐化 + 对比度增强 + 双边滤波 + 二值化
def preprocess_image(image):
    # 转为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 图像锐化处理
    kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpen)

    # 对比度拉伸
    contrast_stretched = cv2.normalize(sharpened, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 双边滤波去噪
    denoised_image = cv2.bilateralFilter(contrast_stretched, d=9, sigmaColor=75, sigmaSpace=75)

    # 二值化处理
    binary_image = cv2.adaptiveThreshold(
        denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return binary_image, denoised_image


# EAST 文本检测并框选出文字区域
def east_text_detection(image):
    model_path = "frozen_east_text_detection.pb"
    net = cv2.dnn.readNet(model_path)

    orig = image.copy()
    (H, W) = image.shape[:2]

    # 设定输入的尺寸
    newW, newH = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # 调整图像尺寸
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # 创建输入blob
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    # 获取输出层
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    (scores, geometry) = net.forward(layerNames)

    # 解码检测结果
    rectangles = []
    confidences = []
    (numRows, numCols) = scores.shape[2:4]
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < 0.5:
                continue

            offsetX, offsetY = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rectangles.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    # 非极大值抑制处理，减少重叠框的数量
    boxes = cv2.dnn.NMSBoxes(rectangles, confidences, score_threshold=0.4, nms_threshold=0.3)

    if len(boxes) > 0:
        for i in boxes.flatten():
            (startX, startY, endX, endY) = rectangles[i]
            # 调整为原图比例
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # 绘制检测到的文本框
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # 返回检测区域的矩形框
            yield (startX, startY, endX, endY, orig)


# 处理图像并执行OCR，保存识别结果到output目录
def process_image_and_recognize(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像：{image_path}")

    # 获取图片文件名
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(output_dir, f"{image_name}.txt")

    # 预处理图像并获得二值化和去噪图像
    binary_image, denoised_image = preprocess_image(image)

    # 使用 EAST 模型框选文字区域
    detected_text = ""
    for (startX, startY, endX, endY, output_image) in east_text_detection(image):
        # 从检测到的区域进行 OCR 识别
        roi = denoised_image[startY:endY, startX:endX]

        # 使用 Tesseract 进行 OCR 识别，并启用 LSTM 模型 --oem 1
        text = pytesseract.image_to_string(roi, lang='chi_sim+eng', config='--oem 1 --psm 6')
        detected_text += text + "\n"

    # 保存识别的完整文本
    save_recognized_text(detected_text, output_filename)

    # 显示处理后的图像和文本框
    cv2.imshow("Text Detection with EAST", output_image)
    cv2.imshow("Binary Image", binary_image)
    cv2.imshow("Denoised Image", denoised_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = 'test.png'
process_image_and_recognize(image_path)
