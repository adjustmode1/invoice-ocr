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
        raw_results = ocr.ocr(img_np, cls=use_angle_cls)

        lines: List[Dict[str, Any]] = []
        full_text_lines: List[str] = []

        # PaddleOCR returns a list of results (one per image/page).
        # For a single image, raw_results[0] contains [[box, (text, score)], ...]
        if raw_results and raw_results[0]:
            for item in raw_results[0]:
                box = item[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                text, score = item[1]

                if score >= confidence_threshold:
                    lines.append({
                        "text": text,
                        "confidence": round(float(score), 4),
                        "box": [[int(pt[0]), int(pt[1])] for pt in box]
                    })
                    full_text_lines.append(text)

        return {
            "image_width": image.width,
            "image_height": image.height,
            "total_detected": len(lines),
            "full_text": "\n".join(full_text_lines),
            "lines": lines
        }
