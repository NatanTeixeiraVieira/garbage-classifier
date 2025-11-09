# Sistema de Detecção de Lixo usando YOLO

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

# Sistema de Detecção de Lixo usando YOLO

Este repositório reúne código e recursos para um sistema de detecção e classificação de lixo baseado em modelos YOLO. O objetivo do projeto é ajudar a classificar diferentes tipos de materiais (como plástico, papel, metal e vidro) a partir de imagens. O projeto contém scripts para treinar modelos, executar inferência via um backend em Python e exemplos de conjuntos de dados organizados por categorias.

## Configuração do Backend Python

## 1. Pré-requisitos

- Python 3.9+ instalado
- Conexão com internet (para download do modelo YOLO)

## 2. Crie um ambiente virtual

python -m venv venv

## 3. Ative o ambiente virtual

### Windows:

venv\Scripts\activate

### Linux/Mac:

source venv/bin/activate

## Instale as dependências

pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install python-multipart==0.0.6
pip install ultralytics==8.0.196
pip install opencv-python==4.8.1.78
pip install Pillow==10.0.1
pip install numpy==1.24.3
pip install pydantic==2.4.2

## 3. Arquivos principais do projeto

main.py - Serve para executar o projeto
train.py - Serve para treinar o modelo
datasets/garbage/train - diretório onde deve ficar o dataset. Ex.:

dataset/
| └──train
| | └── garbage/
| | └── train/
| | ├── battery/
| | ├── biological/
| | ├── brown-glass/
| | ├── cardboard/
| | ├── clothes/
| | ├── green-glass/
| | ├── metal/
| | ├── paper/
| | ├── plastic/
| | ├── shoes/
| | ├── trash/
| | └── white-glass/

Para obter as classes, entrar no dataset baixado, entrar no diretório 'garbage_classification_enhanced', copiar as classes e adicionar dentro de dataset/train

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
