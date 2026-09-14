"""
Script chạy OCR trực tiếp qua dòng lệnh (không cần chạy web server).
Sử dụng:
    python sample_cli.py path/to/invoice.jpg --lang vi
"""

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

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
    try:
        ocr = PaddleOCR(use_angle_cls=not args.no_angle, lang=args.lang, enable_mkldnn=False)
    except (ValueError, TypeError):
        ocr = PaddleOCR(use_angle_cls=not args.no_angle, lang=args.lang)

    print(f"[*] Đang nhận diện văn bản từ: {image_path}...")
    try:
        results = ocr.ocr(str(image_path))
    except TypeError:
        results = ocr.predict(str(image_path))

    extracted_data = []
    if results:
        first_res = results[0]
        res_dict = getattr(first_res, "res", first_res)
        if isinstance(res_dict, dict) and ("dt_polys" in res_dict or "rec_texts" in res_dict or "rec_text" in res_dict):
            dt_polys = res_dict.get("dt_polys", [])
            rec_texts = res_dict.get("rec_texts", res_dict.get("rec_text", []))
            rec_scores = res_dict.get("rec_scores", res_dict.get("rec_score", []))
            print("\n--- KẾT QUẢ NHẬN DIỆN ---")
            for idx, poly in enumerate(dt_polys, 1):
                text = str(rec_texts[idx - 1]) if idx - 1 < len(rec_texts) else ""
                score = float(rec_scores[idx - 1]) if idx - 1 < len(rec_scores) else 1.0
                poly_list = poly.tolist() if hasattr(poly, "tolist") else poly
                print(f"[{idx:02d}] (Độ tin cậy: {score:.2f}) -> {text}")
                extracted_data.append({
                    "text": text,
                    "confidence": round(score, 4),
                    "box": poly_list
                })
        elif isinstance(first_res, (list, tuple)):
            print("\n--- KẾT QUẢ NHẬN DIỆN ---")
            for idx, line in enumerate(first_res, 1):
                if isinstance(line, (list, tuple)) and len(line) == 2:
                    box = line[0]
                    text, score = line[1]
                    print(f"[{idx:02d}] (Độ tin cậy: {score:.2f}) -> {text}")
                    extracted_data.append({
                        "text": str(text),
                        "confidence": round(float(score), 4),
                        "box": box.tolist() if hasattr(box, "tolist") else box
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
