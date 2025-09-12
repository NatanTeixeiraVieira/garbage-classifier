# Sistema de Detecção de Carros usando YOLO

## Descrição do modelo utilizado:

### Yolo (You Only Look Once) v8

### Arquitetura do modelo:

### 1. BackBone:

- CSPDarknet53
- Cross Stage Partial (CSP)
- DarkNet Blocks

### 2. Neck:

- PANet (Path Aggregation Network)
- FPN (Feature Pyramid Network)
- SPP (Spatial Pyramid Pooling)

### 3. Head:

- Anchor-free detection
- Decoupled head
- Multiple output scales

---

### Camadas Principais:

1. Camadas Convolucionais:

- Conv2d + BatchNorm + SiLU Activation
- Filtros: 32, 64, 128, 258, 512, 1024

2. Camadas C2f (CSP Bottleneck)

- Blocs residuais com conexões skip
- Redução de parâmetros mantendo perfomance

3. SPPF (Spatial Pyramind Pooling Fast)

- MaxPool com kernels 5x5
- Concatenação de características multi-escala

4. Upsample Layers:

- Interpolação para fusão de características

---

### Classes de Treinamento (COCO Dataset)

O modelo YOLO utilizado foi treinado com o dataset COCO, para o projeto o qual identifica carro foi utilizado
a classe: **car**.

Além desta classe, possuímos classes relacionadas, como por exemplo: **truck, bus** e **motorcycle**.

#### Parâmetros do modelo:

- **Entrada:** 640x640 pixels
- **Parâmetro:** ~11.2M (YOLOv8n) / ~25.9M (YOLOv8s)
- **GFLOPs:** 8.7 (YOLOv8n) / 28.6 (YOLOv8s)
- **mAP50-95:** 37.3% (YOLOv8n) / 44.9% (YOLOv8s)

---

### Métricas de Performance

1. **Precisão:**

- mAP (mean Average Precision): 44.9% (COCO Dataset)
- Precisão para carros: ~85-90%
- Recall: ~80-85%

2. **Perfomance:**

- Inferência: ~20-50ms por imagem (GPU)
- Throughput: 20-50 FPS dependendo do hardware
- Memória: ~2-4GB RAM para modelo carregado

# Especificações sobre o Backend

- **Backend:** Python em conjunto com o framework **FastAPI** para a API REST.
- **IA:** Ultralytics YOLOv8
- **Processamento:** OpenCV, Pillow
- **Validação:** Pydantic Models

# Configuração do Backend Python

## 1. Pré-requisitos

- Python 3.9+ instalado
- Pelo menos 4GB de RAM disponível
- Conexão com internet (para download do modelo YOLO)

## 2. Crie um ambiente virtual

python -m venv venv

## 3. Ative o ambiente virtual

- ### Windows:

  venv\Scripts\activate

- ### Linux/Mac:
  source venv/bin/activate

## Instale as dependências

- pip install fastapi==0.104.1
- pip install uvicorn==0.24.0
- pip install python-multipart==0.0.6
- pip install ultralytics==8.0.196
- pip install opencv-python==4.8.1.78
- pip install Pillow==10.0.1
- pip install numpy==1.24.3
- pip install pydantic==2.4.2

## 3. Estrutura do Projeto Backend

```
car_detection_backend/
├── main.py
├── requirements.txt
├── models/           # Modelos YOLO serão baixados aqui
└── logs/            # Logs do sistema
```

## 4. Executando o Backend

Execute o servidor

```
python main.py
```

## 5. Testando a API

Teste de status

```
curl http://localhost:8000/
```

## Teste de health check

```
curl http://localhost:8000/health
```

## Teste de upload (caminho da sua imagem)

```
curl.exe -X POST "http://localhost:8000/detect" `
-H "accept: application/json" `
-H "Content-Type: multipart/form-data" `
-F "file=@imagem.jpg"
```

## 6 Errors

This environment is externally managed
╰─> To install Python packages system-wide, try apt install
python3-xyz, where xyz is the package you are trying to
install.
python3.10 -m venv venv

## 7 Rodar sempre

source venv/bin/activate
python3.10 main.py

## 8 Treinar a IA com o dataset

yolo classify train model=yolov8n-cls.pt data=dataset/train epochs=2 imgsz=224
