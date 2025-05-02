
# ocr-hand-writing-digit-2025-0501
# PaddleOCR Flask Web UI (GPU Version)

This project runs PaddleOCR with GPU support using `paddlepaddle-gpu==2.6.2`.

## Requirements

- Python 3.9
- CUDA 12.1 or 12.2
- pip installed

## Install Dependencies

```bash
pip install -r requirements.txt -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

## Run the App

```bash
python app.py
```

Visit http://localhost:5000 to upload images and extract digits using OCR.
