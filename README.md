# CUÁNDOViajo

Predicción de carga en el subte de Buenos Aires. El sistema te dice si el tren va a ir lleno en el horario que elegiste y, si es así, cuándo salir para viajar cómodo.

Desarrollado para la Hackathón Y-Hat · Exactas UBA · 2026.

---

## El problema

El subte de Buenos Aires tiene picos de demanda muy marcados (07:00–09:30 y 17:00–20:00) donde los vagones van saturados. Los datos de molinetes (torniquetes) registran cuántas personas entran en cada estación por intervalo de 15 minutos, pero **no registran salidas**: no sabemos cuánta gente hay en el vagón en un momento dado.

---

## Arquitectura

```
subte_load_prediction.ipynb   → entrena el modelo y genera los artefactos
predictor.py                  → inferencia + fixed-point iteration + ajuste por eventos
news_agent.py                 → eventos del día con factor de ajuste y juicio en lenguaje natural
app.py                        → demo interactiva en Streamlit
CUÁNDOViajo.html              → frontend alternativo (HTML/JS puro, sin servidor)
```

### Estimación de carga (sin datos de salidas)

Se usa la **restricción de terminal**: al llegar a la cabecera, el vagón se vacía. Con eso se calcula una suma rodante de embarques en las *k* estaciones anteriores como proxy de la carga actual. El valor de *k* se optimiza por línea.

### Modelo

XGBoost entrenado con 28 features: temporales (hora, día, bin de 15 min), lags de boardings y carga, rolling means, posición normalizada en la línea y boardings de estaciones vecinas.

| Métrica | Valor |
|---|---|
| MAE | 12.6 pasajeros |
| RMSE | 36.4 |
| MAPE (carga > 20 pax) | 7.4 % |

### Fixed-point iteration

El sistema itera el modelo sobre su propia predicción actualizando `load_lag_1` con el valor predicho, hasta converger. Esto evita el **efecto observador**: si la predicción dice que el tren va lleno y la gente cambia su horario, la predicción se invalida a sí misma.

### Ajuste por eventos

Eventos como recitales, partidos o cortes de servicio no están en los datos históricos. La arquitectura prevista es: scraping de noticias → LLM que evalúa el impacto y ajusta la predicción con un factor multiplicativo y una justificación en lenguaje natural. En el MVP los eventos están hardcodeados con juicios pre-escritos.

### Sugerencia de horario

Si la ocupación predicha es alta, el sistema busca en ±30 minutos el slot más cercano donde la carga sea al menos 25% menor (umbral proporcional). Ejemplo: *"Si salís 15 minutos antes (07:45) vas a viajar mejor: 🟡 Normal"*.

---

## Cómo correr la demo

### Requisitos

```bash
pip install streamlit xgboost pandas numpy
```

### Datos

Los CSVs de molinetes son datos abiertos del Gobierno de la Ciudad de Buenos Aires:
[data.buenosaires.gob.ar](https://data.buenosaires.gob.ar) → buscar *"molinetes subte"*.

Colocarlos en una carpeta `molinetes-2024/` en la raíz del proyecto.

### Entrenar el modelo (opcional)

El modelo ya está entrenado (`xgb_subte_load.json` + `model_meta.pkl`). Para reentrenar, correr `subte_load_prediction.ipynb` en Kaggle con los datos en `/kaggle/input/datasets/axelnahuelbelbrun/subtes`.

### Levantar la app

```bash
streamlit run app.py
```

### Frontend HTML (sin servidor)

```bash
xdg-open CUÁNDOViajo.html
```

---

## Estructura del repositorio

```
├── app.py                        Streamlit app (demo principal)
├── predictor.py                  Carga del modelo, features, fixed-point, predicción
├── news_agent.py                 Eventos hardcodeados con juicios pre-escritos
├── subte_load_prediction.ipynb   Notebook de entrenamiento
├── xgb_subte_load.json           Modelo entrenado (XGBoost native format)
├── model_meta.pkl                Metadata: features, station_sequences, best_k
├── CUÁNDOViajo.html              Frontend alternativo HTML/JS
└── prompt_*.txt                  Prompts usados para desarrollo asistido por IA
```

---

## Limitaciones

- La carga es una **estimación**, no una medición directa (no hay datos de salidas).
- La demo usa datos históricos de 2024; no hay feed en tiempo real.
- El agente de noticias es hardcodeado en el MVP.
- No modela transbordos entre líneas.
