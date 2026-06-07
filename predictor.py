import glob
import os
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

LAGS     = [1, 2, 4, 8, 96]
ROLLS    = [4, 8, 96]
INTERVAL = 15  # minutos por intervalo

# Estado de módulo — se carga una vez con load_model() y lo usan el resto
_MODEL = None
_META  = None


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------

def _build_encodings(station_sequences: dict) -> tuple[dict, dict]:
    """
    Reconstruye los encodings ordinales usados durante el entrenamiento.
    Ambos se basan en orden alfabético, igual que pd.Categorical().codes.
    """
    linea_enc    = {l: i for i, l in enumerate(sorted(station_sequences.keys()))}
    all_stations = sorted({st for seq in station_sequences.values() for st in seq})
    estacion_enc = {s: i for i, s in enumerate(all_stations)}
    return linea_enc, estacion_enc


# ---------------------------------------------------------------------------
# Carga de modelo y datos
# ---------------------------------------------------------------------------

def load_model(model_path: str, meta_path: str):
    """Carga el XGBRegressor y la metadata del pickle. Guarda estado de módulo."""
    global _MODEL, _META
    model = XGBRegressor()
    model.load_model(model_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    _MODEL = model
    _META  = meta
    return model, meta


def load_historical_data(data_path: str, meta: dict, months: int = 3) -> pd.DataFrame:
    """
    Carga y procesa los CSVs de molinetes, devolviendo un DataFrame con
    columnas [TIMESTAMP, LINEA, ESTACION, station_pos, boardings, load_est].

    Parámetros
    ----------
    months : cuántos meses recientes cargar (2 archivos por mes).
             Reducir para acelerar el startup en demo.
    """
    def _read_csv(fp: str) -> pd.DataFrame:
        with open(fp, "r", encoding="utf-8-sig") as f:
            lines = f.read().strip().split("\n")
        rows   = [l.strip().strip('"').split(";") for l in lines if l.strip()]
        header = [h.strip() for h in rows[0]]
        n      = sum(1 for h in header if h)
        data   = [r[:n] for r in rows[1:] if len(r) >= n]
        return pd.DataFrame(data, columns=header[:n])

    files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    files = files[-(months * 2):]          # los más recientes primero por nombre

    raw              = pd.concat([_read_csv(f) for f in files], ignore_index=True)
    raw["FECHA"]     = pd.to_datetime(raw["FECHA"], dayfirst=True, errors="coerce")
    raw["DESDE"]     = pd.to_timedelta(raw["DESDE"], errors="coerce")
    raw["pax_TOTAL"] = pd.to_numeric(raw["pax_TOTAL"], errors="coerce")
    raw["TIMESTAMP"] = raw["FECHA"] + raw["DESDE"]
    raw["ESTACION"]  = raw["ESTACION"].str.strip()
    raw["LINEA"]     = raw["LINEA"].str.strip()
    raw              = raw.dropna(subset=["TIMESTAMP", "pax_TOTAL", "LINEA", "ESTACION"])
    raw["pax_TOTAL"] = raw["pax_TOTAL"].astype(int)

    agg = (
        raw.groupby(["TIMESTAMP", "LINEA", "ESTACION"], as_index=False)["pax_TOTAL"]
        .sum()
        .rename(columns={"pax_TOTAL": "boardings"})
    )

    seq_pos = {
        linea: {st: i for i, st in enumerate(seq)}
        for linea, seq in meta["station_sequences"].items()
    }
    agg["station_pos"] = agg.apply(
        lambda r: seq_pos.get(r["LINEA"], {}).get(r["ESTACION"], -1), axis=1
    )
    agg = agg[agg["station_pos"] >= 0].copy()

    # Calcular load_est con los k óptimos guardados en meta
    best_k     = meta["best_k"]
    load_parts = []
    for linea in agg["LINEA"].unique():
        k   = best_k.get(linea, 5)
        sub = agg[agg["LINEA"] == linea]
        pivot = (
            sub.pivot_table(
                index="TIMESTAMP", columns="station_pos",
                values="boardings", fill_value=0,
            ).sort_index(axis=1)
        )
        load_pivot = pd.DataFrame(0, index=pivot.index, columns=pivot.columns)
        for idx, col in enumerate(pivot.columns):
            start = max(0, idx - k)
            if idx > start:
                load_pivot[col] = pivot.iloc[:, start:idx].sum(axis=1)
        load_long = (
            load_pivot.stack().reset_index().rename(columns={0: "load_est"})
        )
        load_long["LINEA"] = linea
        load_parts.append(load_long)

    estimates = pd.concat(load_parts, ignore_index=True)
    result    = agg.merge(estimates, on=["LINEA", "TIMESTAMP", "station_pos"], how="left")
    result["load_est"] = result["load_est"].fillna(0).astype(int)

    return result.sort_values(["LINEA", "ESTACION", "TIMESTAMP"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Construcción del vector de features
# ---------------------------------------------------------------------------

def build_features(
    linea: str,
    estacion: str,
    timestamp,
    historical_sample: pd.DataFrame,
    meta: dict,
) -> pd.DataFrame:
    """
    Construye el vector de 28 features para una sola observación.
    Para valores sin dato exacto en el histórico se imputa la mediana
    del mismo (dayofweek, time_bin) en el histórico disponible.
    """
    ts    = pd.Timestamp(timestamp)
    seq   = meta["station_sequences"].get(linea, [])
    n_st  = len(seq)
    s_pos = seq.index(estacion) if estacion in seq else 0

    linea_enc, estacion_enc = _build_encodings(meta["station_sequences"])

    # --- Features temporales ---
    hour       = ts.hour
    minute     = ts.minute
    dayofweek  = ts.dayofweek
    month      = ts.month
    is_weekend = int(dayofweek >= 5)
    time_bin   = hour * 4 + minute // 15
    s_norm     = s_pos / n_st if n_st > 0 else 0.0

    # --- Subconjunto histórico para esta (linea, estacion) ---
    hist = (
        historical_sample[
            (historical_sample["LINEA"] == linea) &
            (historical_sample["ESTACION"] == estacion)
        ]
        .set_index("TIMESTAMP")
        .sort_index()
    )

    def _lookup(col: str, lag_n: int) -> float:
        """Valor en ts - lag_n*15min; fallback a mediana del mismo patrón."""
        target = ts - pd.Timedelta(minutes=INTERVAL * lag_n)
        if target in hist.index:
            return float(hist.loc[target, col])
        # Fallback: mediana del mismo (dayofweek, time_bin)
        tb  = target.hour * 4 + target.minute // 15
        dow = target.dayofweek
        mask = (
            (historical_sample["LINEA"] == linea) &
            (historical_sample["ESTACION"] == estacion) &
            (historical_sample["TIMESTAMP"].dt.dayofweek == dow) &
            (
                historical_sample["TIMESTAMP"].dt.hour * 4
                + historical_sample["TIMESTAMP"].dt.minute // 15
                == tb
            )
        )
        vals = historical_sample.loc[mask, col]
        if len(vals) > 0:
            return float(vals.median())
        return float(hist[col].median()) if len(hist) > 0 else 0.0

    boardings = _lookup("boardings", 0)

    # --- Lags ---
    lag_feats: dict = {}
    for lag in LAGS:
        lag_feats[f"b_lag_{lag}"]    = _lookup("boardings", lag)
        lag_feats[f"load_lag_{lag}"] = _lookup("load_est", lag)

    # --- Rolling means ---
    roll_feats: dict = {}
    for win in ROLLS:
        cutoff = ts - pd.Timedelta(minutes=INTERVAL * win)
        window = hist.loc[cutoff:ts, "boardings"]
        window = window[window.index < ts]   # excluir el propio ts
        roll_feats[f"b_roll_{win}"] = float(window.mean()) if len(window) > 0 else boardings

    # --- Features espaciales (estaciones vecinas en el mismo timestamp) ---
    snap = (
        historical_sample[
            (historical_sample["LINEA"] == linea) &
            (historical_sample["TIMESTAMP"] == ts)
        ]
        .set_index("station_pos")
    )

    def _neighbor(offset: int) -> float:
        pos = s_pos + offset
        if pos in snap.index:
            return float(snap.loc[pos, "boardings"])
        return boardings   # fallback: propio valor

    row = {
        "hour": hour, "minute": minute, "dayofweek": dayofweek,
        "month": month, "is_weekend": is_weekend, "time_bin": time_bin,
        "station_pos": s_pos, "station_norm": s_norm,
        "boardings": boardings,
        **lag_feats,
        **roll_feats,
        "b_prev_1st": _neighbor(-1),
        "b_prev_2nd": _neighbor(-2),
        "b_next_1st": _neighbor(1),
        "linea_enc":    linea_enc.get(linea, 0),
        "estacion_enc": estacion_enc.get(estacion, 0),
    }

    return pd.DataFrame([row])[meta["features"]]


# ---------------------------------------------------------------------------
# Punto fijo
# ---------------------------------------------------------------------------

def fixed_point_predict(
    model: XGBRegressor,
    features_row: pd.DataFrame,
    max_iter: int = 10,
    tol: float = 0.5,
) -> tuple[float, list]:
    """
    Itera el modelo sobre su propia predicción actualizando load_lag_1
    hasta que converge (|delta| < tol) o se alcanza max_iter.

    Devuelve (predicción_final, lista_de_valores_por_iteración).
    """
    feat = features_row.copy()
    pred = float(np.maximum(model.predict(feat)[0], 0))
    iterations = [round(pred, 1)]

    for _ in range(max_iter - 1):
        feat = feat.copy()
        feat["load_lag_1"] = pred
        new_pred = float(np.maximum(model.predict(feat)[0], 0))
        iterations.append(round(new_pred, 1))
        if abs(new_pred - pred) < tol:
            pred = new_pred
            break
        pred = new_pred

    return round(pred, 1), iterations


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def predict(
    linea: str,
    estacion: str,
    timestamp,
    historical_sample: pd.DataFrame,
    event_store=None,
    model: XGBRegressor | None = None,
    meta: dict | None = None,
) -> dict:
    """
    Predicción completa: punto fijo + ajuste por evento.

    model y meta son opcionales: si no se pasan se usan los cargados
    por load_model() (estado de módulo). app.py no necesita pasarlos.

    event_store acepta tanto una lista de eventos como el dict completo
    {"events": [...]} que devuelve news_agent.fetch_events().

    Devuelve
    --------
    {
        "base"          : float,        predicción sin ajuste
        "adjusted"      : float,        predicción con factor de evento
        "iterations"    : list[float],  valores por iteración del punto fijo
        "event_applied" : dict | None   evento que matcheó, o None
    }
    """
    model = model or _MODEL
    meta  = meta  or _META
    if model is None or meta is None:
        raise RuntimeError("Llamá a load_model() antes de predict().")

    features         = build_features(linea, estacion, timestamp, historical_sample, meta)
    base, iterations = fixed_point_predict(model, features)

    adjusted      = base
    event_applied = None

    if event_store:
        # Aceptar tanto lista como {"events": [...]}
        events_list = (
            event_store.get("events", [])
            if isinstance(event_store, dict)
            else event_store
        )
        ts_str = pd.Timestamp(timestamp).strftime("%H:%M")
        for event in events_list:
            in_line    = linea    in event.get("lineas_afectadas", [])
            in_station = estacion in event.get("estaciones_afectadas", [])
            in_window  = (
                event.get("ventana_inicio", "00:00") <= ts_str
                <= event.get("ventana_fin", "23:59")
            )
            if in_line and in_station and in_window:
                adjusted      = round(base * event["factor"], 1)
                event_applied = event
                break

    return {
        "base":          base,
        "adjusted":      adjusted,
        "iterations":    iterations,
        "event_applied": event_applied,
    }
