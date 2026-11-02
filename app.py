
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict
import joblib
import pandas as pd
import math
import os

# ==========================================
# 1. CONFIGURACIÓN DE LA API
# ==========================================

app = FastAPI(
    title="API de Clasificación de Cáncer de Mama",
    description="Modelo de aprendizaje automático para fines educativos.",
    version="1.0.0"
)

# ==========================================
# 2. CARGAR EL MODELO
# ==========================================

RUTA_MODELO = "modelo_cancer_mama.joblib"

if not os.path.exists(RUTA_MODELO):
    raise RuntimeError(
        "No se encontró modelo_cancer_mama.joblib. "
        "Coloca el archivo en la carpeta principal."
    )

paquete = joblib.load(RUTA_MODELO)

modelo = paquete["modelo"]
variables = paquete["variables"]
umbral = paquete.get("umbral", 0.50)
clases = paquete.get(
    "clases",
    {0: "Benigno", 1: "Maligno"}
)

# ==========================================
# 3. CONFIGURAR INTERFAZ WEB
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

@app.get("/")
def inicio():
    return FileResponse("static/index.html")

# ==========================================
# 4. ENDPOINT DE SALUD
# ==========================================

@app.get("/health")
def health():
    return {
        "estado": "correcto",
        "modelo_cargado": True
    }

# ==========================================
# 5. CONSULTAR VARIABLES
# ==========================================

@app.get("/features")
def obtener_variables():
    return {
        "variables": variables,
        "total": len(variables)
    }

# ==========================================
# 6. ESTRUCTURA DE DATOS DE ENTRADA
# ==========================================

class DatosPaciente(BaseModel):
    datos: Dict[str, float]

# ==========================================
# 7. ENDPOINT DE PREDICCIÓN
# ==========================================

@app.post("/predict")
def predecir(entrada: DatosPaciente):

    datos = entrada.datos

    faltantes = [
        variable for variable in variables
        if variable not in datos
    ]

    adicionales = [
        variable for variable in datos
        if variable not in variables
    ]

    if faltantes or adicionales:
        raise HTTPException(
            status_code=400,
            detail={
                "variables_faltantes": faltantes,
                "variables_adicionales": adicionales
            }
        )

    try:
        valores = [float(datos[v]) for v in variables]
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Todos los valores deben ser numéricos."
        )

    if not all(math.isfinite(v) for v in valores):
        raise HTTPException(
            status_code=400,
            detail="No se permiten valores infinitos o no válidos."
        )

    X_nuevo = pd.DataFrame(
        [valores],
        columns=variables
    )

    probabilidad = float(
        modelo.predict_proba(X_nuevo)[0][1]
    )

    prediccion = int(probabilidad >= umbral)

    return {
        "prediccion": prediccion,
        "clasificacion": clases[prediccion],
        "probabilidad_maligna": round(probabilidad, 4),
        "umbral": umbral,
        "aviso": (
            "Demostración educativa. "
            "No utilizar para diagnóstico ni decisiones clínicas."
        )
    }
