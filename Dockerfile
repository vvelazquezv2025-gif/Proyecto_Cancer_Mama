
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos de la aplicación
COPY app.py .
COPY static/ ./static/

# Copiar el modelo entrenado
COPY modelo_cancer_mama.joblib .

# Puerto de la aplicación
EXPOSE 8000

# Iniciar API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
