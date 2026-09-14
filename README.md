# Invoice OCR với PaddleOCR & FastAPI (Dockerized)

Dự án OCR cho hóa đơn, chứng từ và tài liệu sử dụng mô hình **PaddleOCR** (hỗ trợ tiếng Việt và tiếng Anh), được đóng gói trong Docker cùng API RESTful viết bằng **FastAPI**.

---

## 📁 Cấu trúc thư mục

```text
invoice-ocr/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI Web Application & Endpoints
│   └── ocr_engine.py        # Logic xử lý và cache mô hình PaddleOCR
├── Dockerfile               # Tối ưu build image & tải sẵn model
├── docker-compose.yml       # Khởi chạy dịch vụ nhanh chóng
├── requirements.txt         # Thư viện Python cần thiết
├── sample_cli.py            # Chạy OCR dòng lệnh trực tiếp (CLI)
├── test_client.py           # Ví dụ gọi API từ Python
└── .dockerignore
```

---

## 🚀 1. Khởi chạy với Docker (Khuyên dùng)

### Cách 1: Sử dụng Docker Compose

```bash
# Build và chạy ngầm
docker compose up -d --build

# Xem logs
docker compose logs -f

# Dừng dịch vụ
docker compose down
```

### Cách 2: Sử dụng Docker CLI thuần

```bash
# 1. Build image (sẽ tự động tải sẵn model tiếng Việt & tiếng Anh)
docker build -t invoice-ocr:latest .

# 2. Chạy container mở port 8000
docker run -d --name invoice_ocr_app -p 8000:8000 invoice-ocr:latest
```

> **Lưu ý:** Dockerfile đã tích hợp sẵn bước **pre-download model** (`vi` và `en`) ngay trong lúc build image. Nhờ đó container khi khởi chạy có thể hoạt động hoàn toàn offline mà không lo lỗi mạng/timeout.

---

## 💻 2. Chạy trực tiếp trên máy cục bộ (Không dùng Docker)

Yêu cầu môi trường Python 3.9 - 3.11.

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv

# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Khởi chạy web server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 3. Tài liệu & Gọi API

Khi server đang chạy, truy cập tài liệu Swagger UI tại:
👉 **`http://localhost:8000/docs`**

### Endpoint 1: Nhận diện văn bản (`POST /ocr`)

#### Gọi bằng cURL:

```bash
curl -X POST "http://localhost:8000/ocr?lang=vi&use_angle_cls=true&confidence_threshold=0.5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/duong_dan_den_anh/hoa_don.jpg"
```

#### Phản hồi JSON mẫu:

```json
{
  "success": true,
  "filename": "hoa_don.jpg",
  "lang": "vi",
  "data": {
    "image_width": 1080,
    "image_height": 1920,
    "total_detected": 2,
    "full_text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nHÓA ĐƠN BÁN HÀNG",
    "lines": [
      {
        "text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
        "confidence": 0.9852,
        "box": [[120, 45], [960, 45], [960, 95], [120, 95]]
      },
      {
        "text": "HÓA ĐƠN BÁN HÀNG",
        "confidence": 0.9912,
        "box": [[350, 120], [730, 120], [730, 170], [350, 170]]
      }
    ]
  }
}
```

### Endpoint 2: Vẽ trực quan khung nhận diện (`POST /ocr/visualize`)

Trả về trực tiếp file ảnh đã vẽ khung đỏ (`bounding box`) quanh các vùng chữ được phát hiện:

```bash
curl -X POST "http://localhost:8000/ocr/visualize?lang=vi" \
  -F "file=@hoa_don.jpg" \
  --output ket_qua_visualize.jpg
```

---

## 🛠️ 4. Chạy trực tiếp qua CLI (`sample_cli.py`)

Nếu bạn muốn test nhanh một ảnh mà không cần bật server:

```bash
python sample_cli.py path/to/hoa_don.jpg --lang vi -o ket_qua.json
```

---

## ⚡ 5. Cấu hình chạy với GPU (Tùy chọn)

Mặc định Dockerfile đang sử dụng phiên bản **CPU** để tương thích trên mọi máy tính. Nếu bạn có **NVIDIA GPU** và muốn tăng tốc:

1. Đổi thư viện trong `requirements.txt`:
   ```text
   paddlepaddle-gpu==2.6.2
   ```
2. Thay base image trong `Dockerfile` thành image có CUDA, ví dụ:
   ```dockerfile
   FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
   # Cài đặt python 3.10 và các gói cần thiết
   ```
3. Cài đặt **NVIDIA Container Toolkit** trên máy host và thêm `deploy.resources.reservations.devices` vào `docker-compose.yml`.
