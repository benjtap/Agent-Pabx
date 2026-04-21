FROM python:3.10-slim

WORKDIR /app

# Installe les dépendances système pour l'audio si nécessaire
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exposer le port pour AudioSocket
EXPOSE 9090

# Commande de lancement
CMD ["python", "main.py"]
