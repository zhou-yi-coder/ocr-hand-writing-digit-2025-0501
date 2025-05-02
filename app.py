import os
import re
import cv2
import numpy as np
import base64
from flask import Flask, request, render_template, jsonify
from paddleocr import PaddleOCR

# 初始化 Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化 PaddleOCR（可改为 use_gpu=True）
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# 将图像区域转为 base64 字符串用于网页展示
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


# 提取数字识别框和对应图像区域
def extract_digits_with_image_blocks(image):
    results = ocr_model.ocr(image, cls=True)
    output = []
    for line in results[0]:
        box, (text, conf) = line
        text = text.strip().replace(" ", "").replace("-", "")
        if not re.match(r'^\d{8,}$', text):
            continue

        # 计算 OCR 框的坐标
        x_coords = [int(pt[0]) for pt in box]
        y_coords = [int(pt[1]) for pt in box]
        x1, x2 = max(min(x_coords), 0), min(max(x_coords), image.shape[1])
        y1, y2 = max(min(y_coords), 0), min(max(y_coords), image.shape[0])

        crop = image[y1:y2, x1:x2]
        encoded = encode_image(crop)

        output.append({
            'image': encoded,
            "box": [[int(x), int(y)] for x, y in box],
            'text': text
        })

    return output


@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    data = request.get_json()
    if not data or 'image_base64' not in data:
        return jsonify({"error": "Missing image_base64"}), 400

    try:
        image_data = base64.b64decode(data['image_base64'])
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400

    results = extract_digits_with_image_blocks(image)
    return jsonify(results)


@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            if image is not None:
                try:
                    results = extract_digits_with_image_blocks(image)
                except Exception as e:
                    print(f"⚠️ OCR failed: {e}")


    return render_template('index.html', results=results)

# 启动应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8801, debug=True)
