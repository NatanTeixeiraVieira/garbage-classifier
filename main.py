from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import logging
import time
import cv2
import os
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationResult(BaseModel):
    class_name: str
    confidence: float

class ApiResponse(BaseModel):
    success: bool
    message: str
    result: Optional[ClassificationResult] = None
    processing_time: Optional[float] = None
    image_dimensions: Optional[List[int]] = None

app = FastAPI(
    title="Garbage Classification API",
    description="API para classificação de tipos de lixo usando YOLOv8",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

class GarbageClassifier:
    def __init__(self):
        try:
            _torch_load = torch.load
            def unsafe_torch_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return _torch_load(*args, **kwargs)
            torch.load = unsafe_torch_load

            # self.model = YOLO("yolov8n-cls.pt")  
            # self.model = YOLO("runs/classify/train7/weights/best.pt")  
            self.model = YOLO("best.pt")  
            logger.info("Modelo YOLO carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo YOLO: {e}")
            raise

    def classify(self, image: np.ndarray) -> dict:
        try:
            start_time = time.time()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model.predict(source=image_rgb)
            top1 = results[0].probs.top1
            class_name = results[0].names[top1]
            confidence = float(results[0].probs.top1conf)
            processing_time = time.time() - start_time
            return {"class_name": class_name, "confidence": confidence, "processing_time": processing_time}
        except Exception as e:
            logger.error(f"Erro na classificação: {e}")
            raise

classifier = GarbageClassifier()

def load_image_from_upload(file: UploadFile) -> np.ndarray:
    try:
        img_bytes = file.file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_array = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception:
        raise HTTPException(status_code=400, detail="Erro ao processar imagem. Verifique se o arquivo é válido.")

@app.get("/")
async def root():
    return {"message": "Garbage Classification API está funcionando!", "status": "online"}

@app.get("/health")
async def health_check():
    try:
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        _ = classifier.model.predict(test_img)
        return {"status": "healthy", "model": "loaded", "timestamp": time.time()}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

@app.post("/classify", response_model=ApiResponse)
async def classify_endpoint(file: UploadFile = File(...)):
    try:
        image = load_image_from_upload(file)
        h, w = image.shape[:2]

        result = classifier.classify(image)

        # object_to_material = {
        #     "water_bottle": "plástico",
        #     "soda_can": "metal",
        #     "wine_glass": "vidro",
        #     "plastic_bag": "plástico",
        #     "cardboard_box": "papel",
        #     "newspaper": "papel",
        #     "tshirt": "tecido",
        #     "shoe": "tecido",
        #     "banana": "orgânico",
        #     "apple": "orgânico",
        #     "battery": "químico",
        #     "aluminum_can": "metal",
        #     "glass_jar": "vidro",
        #     "milk_carton": "papel/plástico",
        #     "cereal_box": "papel",
        # }
        response = ApiResponse(
            success=True,
            message=result['class_name'],
            # message=object_to_material.get(result['class_name'], 'Desconhecido'),
            result=ClassificationResult(class_name=result['class_name'], confidence=result['confidence']),
            processing_time=result['processing_time'],
            image_dimensions=[w, h]
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        return ApiResponse(success=False, message=f"Erro interno: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Erro interno do servidor", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
