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
    Wrapper for PaddleOCR engine with caching, optional preprocessing,
    auto-downscaling for speed, and structured output parsing.
    """
    _instances: Dict[str, PaddleOCR] = {}

    @classmethod
    def get_ocr_instance(
        cls,
        lang: str = "vi",
        use_angle_cls: bool = False,
        use_unwarping: bool = False
    ) -> PaddleOCR:
        key = f"{lang}_{use_angle_cls}_{use_unwarping}"
        if key not in cls._instances:
            logger.info(
                f"Initializing PaddleOCR instance for lang='{lang}', "
                f"angle_cls={use_angle_cls}, unwarping={use_unwarping}..."
            )
            # Try PaddleOCR 3.x with unwarping & orientation toggles
            init_params = {
                "lang": lang,
                "enable_mkldnn": False,
                "use_doc_orientation_classify": use_angle_cls,
                "use_doc_unwarping": use_unwarping,
                "use_textline_orientation": use_angle_cls,
            }
            try:
                cls._instances[key] = PaddleOCR(**init_params)
            except (ValueError, TypeError):
                # Fallback to standard 2.x parameters
                try:
                    cls._instances[key] = PaddleOCR(
                        lang=lang,
                        use_angle_cls=use_angle_cls,
                        enable_mkldnn=False
                    )
                except (ValueError, TypeError):
                    cls._instances[key] = PaddleOCR(lang=lang)

            logger.info(f"PaddleOCR instance [{key}] initialized successfully.")
        return cls._instances[key]

    @classmethod
    def process_image(
        cls,
        image_bytes: bytes,
        lang: str = "vi",
        use_angle_cls: bool = False,
        use_unwarping: bool = False,
        confidence_threshold: float = 0.0,
        max_side_len: int = 1500
    ) -> Dict[str, Any]:
        """
        Processes an image from bytes:
        - Auto-downscales if larger than max_side_len (drastically reduces CPU compute)
        - Maps bounding boxes back to original coordinates
        - Skips expensive unwarping/angle models unless explicitly requested
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.width, image.height

        # Auto-downscale large images (e.g. 4000x3000 -> 1500x1125) to speed up OCR by 3-5x
        scale = 1.0
        if max_side_len and max_side_len > 0:
            max_dim = max(orig_w, orig_h)
            if max_dim > max_side_len:
                scale = max_side_len / float(max_dim)
                new_w = max(1, int(orig_w * scale))
                new_h = max(1, int(orig_h * scale))
                image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
                logger.info(f"Downscaled image from {orig_w}x{orig_h} to {new_w}x{new_h} (scale={scale:.3f})")

        img_np = np.array(image)
        ocr = cls.get_ocr_instance(
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_unwarping=use_unwarping
        )

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
                        # Scale coordinates back to original image size
                        scaled_box = [[int(round(pt[0] / scale)), int(round(pt[1] / scale))] for pt in poly_list]
                        lines.append({
                            "text": text,
                            "confidence": round(score, 4),
                            "box": scaled_box
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
                            scaled_box = [[int(round(pt[0] / scale)), int(round(pt[1] / scale))] for pt in box_list]
                            lines.append({
                                "text": str(text),
                                "confidence": round(float(score), 4),
                                "box": scaled_box
                            })
                            full_text_lines.append(str(text))

        return {
            "image_width": orig_w,
            "image_height": orig_h,
            "processed_width": image.width,
            "processed_height": image.height,
            "scale_factor": round(scale, 4),
            "total_detected": len(lines),
            "full_text": "\n".join(full_text_lines),
            "lines": lines
        }
