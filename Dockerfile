# Sử dụng Python 3.10 slim để tối ưu kích thước image
FROM python:3.10-slim

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    FLAGS_use_mkldnn=0 \
    PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0 \
    FLAGS_enable_pir_in_executor=0

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements và cài đặt dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Tải trước (pre-download) model PaddleOCR cho tiếng Việt và tiếng Anh trong lúc build image
# Giúp container khởi động nhanh và chạy được cả trong môi trường offline/không có internet
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='vi'); PaddleOCR(use_angle_cls=True, lang='en')"

# Copy mã nguồn vào container
COPY app/ ./app

# Mở port 8000 cho FastAPI
EXPOSE 8000

# Lệnh khởi chạy ứng dụng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
