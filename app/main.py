import logging
import io
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw, ImageFont

from app.ocr_engine import OCREngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

app = FastAPI(
    title="Invoice OCR API (PaddleOCR)",
    description="REST API for Document & Invoice OCR using PaddleOCR (Optimized)",
    version="1.1.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Pre-warming fast Core OCR Engine (det + rec) with default language 'vi'...")
    try:
        OCREngine.get_ocr_instance(lang="vi", use_angle_cls=False, use_unwarping=False)
        logger.info("Core OCR Engine pre-warmed successfully.")
    except Exception as e:
        logger.warning(f"Failed to pre-warm OCR model on startup: {e}")

@app.get("/")
def index():
    return {
        "service": "Invoice OCR API",
        "backend": "PaddleOCR",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ocr", summary="Extract text and bounding boxes from image (Fast & Optimized)")
async def extract_ocr(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WEBP, etc.)"),
    lang: str = Query("vi", description="Language code: 'vi', 'en', 'ch', etc."),
    use_angle_cls: bool = Query(False, description="Enable orientation classification (slower)"),
    use_unwarping: bool = Query(False, description="Enable document unwarping/dewarping (slower)"),
    max_side_len: int = Query(1500, description="Auto-downscale max dimension to speed up OCR (0 to disable)"),
    confidence_threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File uploaded is not a valid image format: {file.content_type}"
        )

    try:
        contents = await file.read()
        results = OCREngine.process_image(
            image_bytes=contents,
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_unwarping=use_unwarping,
            confidence_threshold=confidence_threshold,
            max_side_len=max_side_len
        )
        return {
            "success": True,
            "filename": file.filename,
            "lang": lang,
            "data": results
        }
    except Exception as e:
        logger.error(f"Error processing OCR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/ocr/visualize", summary="Extract OCR and return image with rendered bounding boxes")
async def visualize_ocr(
    file: UploadFile = File(..., description="Image file"),
    lang: str = Query("vi"),
    use_angle_cls: bool = Query(False),
    use_unwarping: bool = Query(False),
    max_side_len: int = Query(1500)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file format")

    try:
        contents = await file.read()
        results = OCREngine.process_image(
            image_bytes=contents,
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_unwarping=use_unwarping,
            max_side_len=max_side_len
        )

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        draw = ImageDraw.Draw(image)

        for line in results["lines"]:
            box = [tuple(pt) for pt in line["box"]]
            draw.polygon(box, outline="red", width=2)

        output_io = io.BytesIO()
        image.save(output_io, format="JPEG")
        output_io.seek(0)

        return Response(content=output_io.getvalue(), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error visualizing OCR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")
