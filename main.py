from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import logging
import time
import os
import torch

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos Pydantic para validação
class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class ApiResponse(BaseModel):
    success: bool
    message: str
    detections: Optional[List[DetectionResult]] = None
    processing_time: Optional[float] = None
    image_dimensions: Optional[List[int]] = None

# Inicialização da aplicação
app = FastAPI(
    title="Car Detection API",
    description="API para detecção de carros usando YOLOv8",
    version="1.0.0"
)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações globais
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
CONFIDENCE_THRESHOLD = 0.5
CAR_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

class CarDetector:
    def __init__(self):
        try:
            _torch_load = torch.load
            def unsafe_torch_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return _torch_load(*args, **kwargs)

            torch.load = unsafe_torch_load

            self.model = YOLO('yolov8n.pt')
            # self.model = torch.load('yolov8n.pt', weights_only=False)  # Usa modelo nano para velocidade
            logger.info("Modelo YOLO carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo YOLO: {e}")
            raise

    def detect_cars(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detecta carros na imagem usando YOLO
        """
        try:
            start_time = time.time()
            
            # Executa detecção
            # results = self.model.predict(source=image, conf=CONFIDENCE_THRESHOLD)

            results = self.model(image, conf=CONFIDENCE_THRESHOLD)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extrai informações da detecção
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Filtra apenas classes de veículos
                        if class_name.lower() in [c.lower() for c in CAR_CLASSES]:
                            # Coordenadas da bounding box
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            detection = DetectionResult(
                                class_name=class_name,
                                confidence=confidence,
                                bbox=[x1, y1, x2, y2]
                            )
                            detections.append(detection)
            
            processing_time = time.time() - start_time
            logger.info(f"Detecção concluída em {processing_time:.3f}s. Encontrados {len(detections)} carros")
            
            return detections, processing_time
            
        except Exception as e:
            logger.error(f"Erro na detecção: {e}")
            raise

# Instância global do detector
detector = CarDetector()

def validate_image_file(file: UploadFile) -> None:
    """
    Valida o arquivo de imagem
    """
    # Verifica extensão
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado. Use: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Verifica tamanho
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Arquivo muito grande. Máximo: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )

def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """
    Carrega e processa imagem do upload
    """
    try:
        # Lê bytes do arquivo
        image_bytes = file.file.read()
        
        # Converte para PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Converte para RGB se necessário
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Converte para numpy array (OpenCV format)
        image_array = np.array(pil_image)
        
        # Converte RGB para BGR (formato OpenCV)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
        
    except Exception as e:
        logger.error(f"Erro ao processar imagem: {e}")
        raise HTTPException(
            status_code=400,
            detail="Erro ao processar imagem. Verifique se o arquivo é uma imagem válida."
        )

@app.get("/")
async def root():
    """Endpoint de status da API"""
    return {"message": "Car Detection API está funcionando!", "status": "online"}

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    try:
        # Testa se o modelo está funcionando
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = detector.model(test_image, conf=0.9)
        
        return {
            "status": "healthy",
            "model": "loaded",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check falhou: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/detect", response_model=ApiResponse)
async def detect_cars_endpoint(file: UploadFile = File(...)):
    """
    Endpoint principal para detecção de carros
    """
    try:
        logger.info(f"Recebida requisição de detecção: {file.filename}")
        
        # Validações
        validate_image_file(file)
        
        # Carrega imagem
        image = load_image_from_upload(file)
        image_height, image_width = image.shape[:2]
        
        # Executa detecção
        detections, processing_time = detector.detect_cars(image)
        
        # Prepara resposta
        response = ApiResponse(
            success=True,
            message=f"Detecção concluída. Encontrados {len(detections)} carros.",
            detections=detections,
            processing_time=processing_time,
            image_dimensions=[image_width, image_height]
        )
        
        logger.info(f"Resposta enviada: {len(detections)} detecções")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro interno: {e}")
        return ApiResponse(
            success=False,
            message=f"Erro interno do servidor: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler global de exceções"""
    logger.error(f"Erro não tratado: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Erro interno do servidor",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")