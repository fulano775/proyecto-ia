# Proyecto de IA

Estructura base para crear, entrenar y usar un modelo de Inteligencia Artificial en Python.

## Estructura

```
proyecto_ia/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # Datos originales (no se suben a git)
│   └── processed/    # Datos limpios / transformados
├── models/           # Modelos entrenados (.pt)
├── notebooks/        # Experimentos y análisis exploratorio
├── src/
│   ├── __init__.py
│   ├── config.py     # Hiperparámetros y rutas
│   ├── data.py       # Carga y preprocesado de datos
│   ├── model.py      # Arquitectura de la red neuronal
│   ├── train.py      # Bucle de entrenamiento
│   └── predict.py    # Inferencia con el modelo entrenado
└── tests/
    └── test_model.py
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

1. Entrenar el modelo:
```bash
python -m src.train
```

2. Hacer predicciones:
```bash
python -m src.predict
```

3. Ejecutar pruebas:
```bash
pytest tests/
```

## Próximos pasos

- Reemplazar el dataset de ejemplo por tus datos en `data/raw/`.
- Ajustar la arquitectura en `src/model.py`.
- Registrar experimentos (MLflow, Weights & Biases, etc.).
