"""
Script chạy OCR trực tiếp qua dòng lệnh (không cần chạy web server).
Sử dụng:
    python sample_cli.py path/to/invoice.jpg --lang vi
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

def main():
    parser = argparse.ArgumentParser(description="Chạy OCR trên ảnh hóa đơn/tài liệu bằng PaddleOCR")
    parser.add_argument("image_path", type=str, help="Đường dẫn tới file ảnh cần nhận diện")
    parser.add_argument("--lang", type=str, default="vi", help="Ngôn ngữ nhận diện: vi, en, ch,... (mặc định: vi)")
    parser.add_argument("--no-angle", action="store_true", help="Tắt tính năng xoay và phân loại góc văn bản")
    parser.add_argument("--output", "-o", type=str, default=None, help="Đường dẫn lưu kết quả JSON (tùy chọn)")
    
    args = parser.parse_args()
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        print(f"Lỗi: Không tìm thấy file tại '{image_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Đang khởi tạo PaddleOCR (ngôn ngữ: {args.lang})...")
    ocr = PaddleOCR(use_angle_cls=not args.no_angle, lang=args.lang)

    print(f"[*] Đang nhận diện văn bản từ: {image_path}...")
    results = ocr.ocr(str(image_path), cls=not args.no_angle)

    extracted_data = []
    if results and results[0]:
        print("\n--- KẾT QUẢ NHẬN DIỆN ---")
        for idx, line in enumerate(results[0], 1):
            box = line[0]
            text, score = line[1]
            print(f"[{idx:02d}] (Độ tin cậy: {score:.2f}) -> {text}")
            extracted_data.append({
                "text": text,
                "confidence": round(float(score), 4),
                "box": box
            })
    else:
        print("Không phát hiện được đoạn văn bản nào trong ảnh.")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        print(f"\n[+] Đã lưu kết quả vào: {output_path}")

if __name__ == "__main__":
    main()
