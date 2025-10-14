FROM python:3.9.24-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt main.py best.pt  ./

RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# RUN rm -rf dataset runs

# EXPOSE 8000

CMD ["python", "main.py"]