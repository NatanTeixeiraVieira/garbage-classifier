# Sistema de classificação de lixo com YOLO

## 1. Pré-requisitos

- Python 3.9+ instalado
- Pelo menos 4GB de RAM disponível
- Conexão com internet (para download do modelo YOLO)
- O exeplos a seguir são com python <comando>, mas pode ser utilizado a versão do python instalado junto. A versão que funciona corretamente é python3.10

## 2. Crie um ambiente virtual

python -m venv venv

## 3. Ative o ambiente virtual

- ### Windows:

  venv\Scripts\activate

- ### Linux/Mac:
  source venv/bin/activate

## 4. Instale as dependências

pip install -r requirements.txt

## 5. Executando o Backend

Execute o servidor

```
python main.py
```

## 6. Testando a API

Teste de status

```
curl http://localhost:8000/
```

### Teste de health check

```
curl http://localhost:8000/health
```

### Teste de upload (caminho da sua imagem)

```
curl -X POST "http://localhost:8000/classify" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@metal.jpg"
```

## Erros

This environment is externally managed
╰─> To install Python packages system-wide, try apt install
python3-xyz, where xyz is the package you are trying to
install.
python -m venv venv

## Executar sempre que for rodar o projeto

### Ativar a venv

source venv/bin/activate

### Rodar o projeto

python main.py

## Dataset

- Dataset: Garbage Classification (12 classes) ENHANCED (Kaggle)
- Link: https://www.kaggle.com/datasets/huberthamelin/garbage-classification-labels-corrections
- Para rodar, baixar o dataset e deixar as 12 classes dentro do diretório dataset/train, na raíz do projeto

## Medidas para prevenção de erros ao rodar

- Em caso de problemas ao rodar o projeto:

1. Instalar e configurar o Docker
2. Baixar a imagem: docker pull natanteixeiravieira/garbage-classifier-api
3. Rodar: docker run -itd -p 8000:8000 --name garbage-classifier-api natanteixeiravieira/garbage-classifier-api
   Com isso, o projeto roda na porta 8000 normalmente
