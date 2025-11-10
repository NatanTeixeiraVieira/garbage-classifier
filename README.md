# Sistema de Detecção de Lixo usando YOLO

# Rodar o projeoto (Recomendado)

- Instalar o docker
- Rodar

```bash
  docker pull natanteixeiravieira/garbage-classifier-api:v2.0.0
  docker run -itd -p 8000:8000 --name garbage-classifier-api natanteixeiravieira/garbage-classifier-api:v2.0.0
```

# Configuração do Backend Python

## 1. Pré-requisitos

- Python 3.9+ instalado
- Conexão com internet (para download do modelo YOLO)

## 2. Crie um ambiente virtual

python -m venv venv

## 3. Ative o ambiente virtual

- ### Windows:

  venv\Scripts\activate

- ### Linux/Mac:
  source venv/bin/activate

## Instale as dependências

pip install --no-cache-dir -r requirements.txt

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

pip install --no-cache-dir -r requirements.txt

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

'imagem.jpg' é o nome da imagem. Trocar pelo caminho da imagem de teste

```
curl -X POST "http://localhost:8000/classify" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@imagem.jpg"
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
