"""
Ví dụ gọi API Invoice OCR từ Python (yêu cầu thư viện requests: pip install requests)
"""

import sys
import json
import requests
from pathlib import Path

def test_ocr_api(image_path: str, server_url: str = "http://localhost:8000"):
    endpoint = f"{server_url}/ocr"
    file_path = Path(image_path)
    
    if not file_path.exists():
        print(f"Lỗi: Không tìm thấy file {image_path}")
        return

    print(f"[*] Đang gửi request tới: {endpoint}")
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "image/jpeg")}
        params = {
            "lang": "vi",
            "use_angle_cls": True,
            "confidence_threshold": 0.5
        }
        response = requests.post(endpoint, files=files, params=params)

    if response.status_code == 200:
        data = response.json()
        print("\n[+] Gọi API thành công!")
        print(f"Tổng số dòng văn bản: {data['data']['total_detected']}")
        print("\n--- TOÀN BỘ VĂN BẢN (FULL TEXT) ---")
        print(data['data']['full_text'])
        print("\n--- CHI TIẾT TỪNG DÒNG ---")
        for line in data['data']['lines']:
            print(f"- ({line['confidence']:.2f}) {line['text']}")
    else:
        print(f"[-] Lỗi {response.status_code}: {response.text}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_ocr_api(sys.argv[1])
    else:
        print("Sử dụng: python test_client.py <duong_dan_file_anh>")
