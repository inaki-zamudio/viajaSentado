"""
app.py
------
MVP de predicción de carga en el subte de Buenos Aires.
Streamlit app — ejecutar con: streamlit run app.py
"""

import datetime
import os
import streamlit as st
import pandas as pd

import predictor
import news_agent

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ViajaSentado — Predicción de carga en subte",
    page_icon="🚇",
    layout="wide",
)

st.title("ViajaSentado — Predicción de carga en subte")

# ---------------------------------------------------------------------------
# Inicialización de session_state
# ---------------------------------------------------------------------------
if "events" not in st.session_state:
    st.session_state["events"] = []
if "last_update" not in st.session_state:
    st.session_state["last_update"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_suggestion" not in st.session_state:
    st.session_state["last_suggestion"] = None

# ---------------------------------------------------------------------------
# Carga de modelo y datos (una sola vez)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_model():
    model, meta = predictor.load_model("xgb_subte_load.json", "model_meta.pkl")
    return model, meta


@st.cache_data
def init_data(_meta):
    data_path = "molinetes-2024/" if os.path.isdir("molinetes-2024/") else "data_sample/"
    return predictor.load_historical_data(data_path, _meta)


# ---------------------------------------------------------------------------
# Helpers de UI
# ---------------------------------------------------------------------------

def occupancy_label(pax: float) -> tuple[str, str]:
    if pax < 40:
        return "🟢 Tranquilo", "green"
    elif pax < 80:
        return "🟡 Normal", "orange"
    elif pax < 130:
        return "🟠 Ocupado", "orange"
    else:
        return "🔴 Saturado", "red"


def confidence_label(iterations: list, event_applied: dict | None) -> str:
    n = len(iterations)
    if n <= 2:
        level = "Alta" if not event_applied else "Media"
    elif n <= 5:
        level = "Media" if not event_applied else "Baja"
    else:
        level = "Baja"
    icons = {"Alta": "✅", "Media": "⚠️", "Baja": "❌"}
    return f"{icons[level]} {level}"


SUGGEST_IMPROVEMENT = 0.25  # mejora mínima requerida (proporción de la carga actual)


def find_better_time(linea, estacion, base_ts, historical_sample, event_store, current_load):
    """
    Busca el slot más cercano en ±30 min donde la carga sea al menos
    SUGGEST_IMPROVEMENT menor que current_load (umbral proporcional, no fijo).
    Devuelve (offset_minutos, "antes"|"después", timestamp, carga_estimada) o None.
    """
    target = current_load * (1 - SUGGEST_IMPROVEMENT)
    candidates = []
    for sign, direction in [(-1, "antes"), (1, "después")]:
        for off in range(15, 31, 15):
            ts_c  = base_ts + datetime.timedelta(minutes=sign * off)
            t_str = ts_c.strftime("%H:%M")
            dow   = ts_c.weekday()
            if not (_OPENING[dow] <= t_str <= _CLOSING.get(linea, {}).get(dow, "23:00")):
                continue
            try:
                r = predictor.predict(
                    linea=linea, estacion=estacion, timestamp=ts_c,
                    historical_sample=historical_sample, event_store=event_store,
                )
                if r["adjusted"] < target:
                    candidates.append((off, direction, ts_c, r["adjusted"]))
                    break
            except Exception:
                pass
    return min(candidates, key=lambda x: x[0]) if candidates else None


# Horarios de servicio — apertura idéntica para todas las líneas,
# cierre = último despacho desde la cabecera más cercana (máximo de ambas terminales).
# LineaH sábados: 00:20 real, se acota a 23:45 para evitar manejo de día siguiente.
_OPENING = {
    0: "05:30", 1: "05:30", 2: "05:30", 3: "05:30", 4: "05:30",
    5: "06:00",
    6: "08:00",
}
_CLOSING = {
    "LineaA": {0:"23:28", 1:"23:28", 2:"23:28", 3:"23:28", 4:"23:28", 5:"23:57", 6:"22:36"},
    "LineaB": {0:"23:30", 1:"23:30", 2:"23:30", 3:"23:30", 4:"23:30", 5:"23:53", 6:"22:28"},
    "LineaC": {0:"23:33", 1:"23:33", 2:"23:33", 3:"23:33", 4:"23:33", 5:"23:54", 6:"22:34"},
    "LineaD": {0:"23:28", 1:"23:28", 2:"23:28", 3:"23:28", 4:"23:28", 5:"23:52", 6:"22:28"},
    "LineaE": {0:"23:30", 1:"23:30", 2:"23:30", 3:"23:30", 4:"23:30", 5:"23:58", 6:"22:28"},
    "LineaH": {0:"23:51", 1:"23:51", 2:"23:51", 3:"23:51", 4:"23:51", 5:"23:45", 6:"22:51"},
}

model, meta = init_model()
historical_df = init_data(meta)

station_sequences: dict = meta["station_sequences"]
LINEAS = sorted(station_sequences.keys())

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Parámetros")

    linea = st.selectbox("Línea", LINEAS)
    estaciones = station_sequences.get(linea, [])
    estacion = st.selectbox("Estación", estaciones)

    fecha = st.date_input(
        "Fecha",
        value=datetime.date(2024, 6, 1),
        min_value=datetime.date(2024, 1, 1),
        max_value=datetime.date(2024, 12, 31),
    )

    dow        = fecha.weekday()
    open_str   = _OPENING[dow]
    close_str  = _CLOSING.get(linea, {}).get(dow, "23:00")
    all_times  = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]
    time_options = [t for t in all_times if open_str <= t <= close_str]
    default_t  = "08:00" if "08:00" in time_options else time_options[0]
    hora_str   = st.selectbox("Horario", time_options, index=time_options.index(default_t))

    btn_predecir = st.button("Predecir", type="primary", use_container_width=True)

    st.divider()

    btn_noticias = st.button("Actualizar noticias", use_container_width=True)

# ---------------------------------------------------------------------------
# Acción: actualizar noticias
# ---------------------------------------------------------------------------
if btn_noticias:
    with st.spinner("Buscando eventos del día en Buenos Aires..."):
        try:
            result = news_agent.fetch_events()
            st.session_state["events"] = result.get("events", [])
            st.session_state["last_update"] = datetime.datetime.now()
        except Exception as e:
            st.warning(f"No se pudieron obtener noticias: {e}")

# ---------------------------------------------------------------------------
# Acción: predecir
# ---------------------------------------------------------------------------
if btn_predecir:
    hora, minuto = int(hora_str[:2]), int(hora_str[3:])
    ts = datetime.datetime(fecha.year, fecha.month, fecha.day, hora, minuto)

    mask = (
        (historical_df["LINEA"] == linea) &
        (historical_df["ESTACION"] == estacion)
    )
    historical_sample = historical_df[mask].copy()

    event_store = (
        {"events": st.session_state["events"]}
        if st.session_state["events"]
        else None
    )

    try:
        result = predictor.predict(
            linea=linea,
            estacion=estacion,
            timestamp=ts,
            historical_sample=historical_sample,
            event_store=event_store,
        )
        st.session_state["last_result"] = result
        occ_now, _ = occupancy_label(result["adjusted"])
        if "Tranquilo" not in occ_now:
            st.session_state["last_suggestion"] = find_better_time(
                linea, estacion, ts, historical_sample, event_store,
                current_load=result["adjusted"],
            )
        else:
            st.session_state["last_suggestion"] = None
    except Exception as e:
        st.error(f"Error al predecir: {e}")

# ---------------------------------------------------------------------------
# Panel principal — Sección 1: resultado de predicción
# ---------------------------------------------------------------------------
st.subheader("Predicción de carga")

if st.session_state["last_result"] is not None:
    res      = st.session_state["last_result"]
    base     = res["base"]
    adjusted = res["adjusted"]
    iters    = res["iterations"]
    event    = res["event_applied"]

    occ_text, _ = occupancy_label(adjusted)
    conf_text   = confidence_label(iters, event)

    st.markdown(f"#### {linea} · {estacion}")
    st.markdown(f"# {occ_text}")
    st.caption(f"~{adjusted:.0f} pasajeros estimados")
    st.caption(f"Confiabilidad: **{conf_text}**")

    if event is not None:
        delta_pct = (adjusted - base) / max(base, 1) * 100
        st.caption(
            f"Sin evento: ~{base:.0f} pax  ·  "
            f"Ajuste: {delta_pct:+.1f}% por evento"
        )
        with st.container(border=True):
            tipo_icon = "📈" if event["tipo"] == "aumento" else "📉"
            st.markdown(f"**{tipo_icon} Evento aplicado**")
            st.write(event["descripcion"])
            st.caption(
                f"Factor: **×{event['factor']}** · "
                f"{event['ventana_inicio']} – {event['ventana_fin']}"
            )

    suggestion = st.session_state.get("last_suggestion")
    if suggestion:
        off_min, direction, ts_better, load_better = suggestion
        occ_better, _ = occupancy_label(load_better)
        hora_mejor = ts_better.strftime("%H:%M")
        st.info(
            f"💡 **Si salís {off_min} minutos {direction}** ({hora_mejor}) "
            f"vas a viajar mejor: {occ_better} (~{load_better:.0f} pasajeros)"
        )
else:
    st.info("Seleccioná los parámetros y presioná **Predecir** para ver el resultado.")

# ---------------------------------------------------------------------------
# Panel principal — Sección 2: eventos activos
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Eventos activos")

if st.session_state["last_update"] is not None:
    ts_str = st.session_state["last_update"].strftime("%d/%m/%Y %H:%M:%S")
    st.caption(f"Última actualización: {ts_str}")

events = st.session_state["events"]

if not events:
    st.info("No se encontraron eventos relevantes hoy.")
else:
    for ev in events:
        tipo_icon  = "📈" if ev.get("tipo") == "aumento" else "📉"
        lineas_str = ", ".join(ev.get("lineas_afectadas", []))
        estacs_str = ", ".join(ev.get("estaciones_afectadas", []))
        ventana    = f"{ev.get('ventana_inicio', '?')} – {ev.get('ventana_fin', '?')}"
        factor     = ev.get("factor", 1.0)

        with st.container(border=True):
            cols = st.columns([3, 1, 1, 1])
            cols[0].markdown(f"**{tipo_icon} {ev.get('descripcion', '')}**")
            cols[1].markdown(f"**Líneas**\n{lineas_str or '—'}")
            cols[2].markdown(f"**Horario**\n{ventana}")
            cols[3].markdown(f"**Factor**\n×{factor}")

            if estacs_str:
                st.caption(f"Estaciones afectadas: {estacs_str}")
