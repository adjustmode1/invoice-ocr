import os
# Fix CPU oneDNN / PIR instruction bug in PaddlePaddle 3.x
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

import io
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

logger = logging.getLogger("ocr_engine")

class OCREngine:
    """
    Wrapper for PaddleOCR engine with caching and structured output parsing.
    """
    _instances: Dict[str, PaddleOCR] = {}

    @classmethod
    def get_ocr_instance(cls, lang: str = "vi", use_angle_cls: bool = True) -> PaddleOCR:
        key = f"{lang}_{use_angle_cls}"
        if key not in cls._instances:
            logger.info(f"Initializing PaddleOCR instance for lang='{lang}', use_angle_cls={use_angle_cls}...")
            # Try initializing with enable_mkldnn=False to bypass oneDNN CPU bug
            try:
                cls._instances[key] = PaddleOCR(
                    use_angle_cls=use_angle_cls,
                    lang=lang,
                    enable_mkldnn=False
                )
            except (ValueError, TypeError):
                cls._instances[key] = PaddleOCR(
                    use_angle_cls=use_angle_cls,
                    lang=lang
                )
            logger.info("PaddleOCR initialization completed.")
        return cls._instances[key]

    @classmethod
    def process_image(
        cls,
        image_bytes: bytes,
        lang: str = "vi",
        use_angle_cls: bool = True,
        confidence_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Processes an image from bytes and returns structured OCR results.
        """
        # Load image with PIL and convert to numpy array (RGB)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)

        ocr = cls.get_ocr_instance(lang=lang, use_angle_cls=use_angle_cls)
        try:
            raw_results = ocr.ocr(img_np)
        except TypeError:
            raw_results = ocr.predict(img_np)

        lines: List[Dict[str, Any]] = []
        full_text_lines: List[str] = []

        if raw_results:
            first_res = raw_results[0]
            # Check for PaddleOCR 3.x / PaddleX format
            res_dict = getattr(first_res, "res", first_res)
            if isinstance(res_dict, dict) and ("dt_polys" in res_dict or "rec_texts" in res_dict or "rec_text" in res_dict):
                dt_polys = res_dict.get("dt_polys", [])
                rec_texts = res_dict.get("rec_texts", res_dict.get("rec_text", []))
                rec_scores = res_dict.get("rec_scores", res_dict.get("rec_score", []))

                for i, poly in enumerate(dt_polys):
                    text = str(rec_texts[i]) if i < len(rec_texts) else ""
                    score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                    if score >= confidence_threshold:
                        poly_list = poly.tolist() if hasattr(poly, "tolist") else poly
                        lines.append({
                            "text": text,
                            "confidence": round(score, 4),
                            "box": [[int(pt[0]), int(pt[1])] for pt in poly_list]
                        })
                        full_text_lines.append(text)
            # Check for classic PaddleOCR 2.x format: list of [[box, (text, score)], ...]
            elif isinstance(first_res, (list, tuple)):
                for item in first_res:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        box = item[0]
                        text, score = item[1]
                        if score >= confidence_threshold:
                            box_list = box.tolist() if hasattr(box, "tolist") else box
                            lines.append({
                                "text": str(text),
                                "confidence": round(float(score), 4),
                                "box": [[int(pt[0]), int(pt[1])] for pt in box_list]
                            })
                            full_text_lines.append(str(text))

        return {
            "image_width": image.width,
            "image_height": image.height,
            "total_detected": len(lines),
            "full_text": "\n".join(full_text_lines),
            "lines": lines
        }
