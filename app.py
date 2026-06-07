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
import plotly.graph_objects as go

import predictor
import news_agent

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CUÁNDOViajo — Predicción de carga en subte",
    page_icon="logo.svg",
    layout="wide",
)

st.logo("logo.svg")

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("logo.svg", width=90)
with col_title:
    st.title("CUÁNDOViajo")
    st.caption("Predicción de carga en el subte de Buenos Aires")

st.markdown(
    "Elegí una línea, estación y horario para saber si el tren va a ir lleno. "
    "Si hay un momento cercano con menos gente, te lo decimos."
)
st.caption("Entrenado con datos de molinetes SBASE 2024 · Modelo XGBoost · Sin feed en tiempo real")

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




SUGGEST_IMPROVEMENT = 0.25  # mejora mínima requerida (proporción de la carga actual)
_OCC_RANK = {"🟢 Tranquilo": 0, "🟡 Normal": 1, "🟠 Ocupado": 2, "🔴 Saturado": 3}


def find_better_time(linea, estacion, base_ts, historical_sample, event_store, current_load):
    """
    Busca el slot más cercano en ±30 min donde la carga sea al menos
    SUGGEST_IMPROVEMENT menor Y en una categoría de ocupación mejor.
    Devuelve (offset_minutos, "antes"|"después", timestamp, carga_estimada) o None.
    """
    target       = current_load * (1 - SUGGEST_IMPROVEMENT)
    current_rank = _OCC_RANK.get(occupancy_label(current_load)[0], 3)
    candidates   = []
    for sign, direction in [(-1, "antes"), (1, "después")]:
        for off in range(15, 31, 15):
            ts_c  = base_ts + datetime.timedelta(minutes=sign * off)
            t_str = ts_c.strftime("%H:%M")
            dow   = ts_c.weekday()
            if not (_OPENING[dow] <= t_str <= _CLOSING.get(linea, {}).get(dow, "23:00")):
                continue
            try:
                r            = predictor.predict(
                    linea=linea, estacion=estacion, timestamp=ts_c,
                    historical_sample=historical_sample, event_store=event_store,
                )
                occ_c, _     = occupancy_label(r["adjusted"])
                better_cat   = _OCC_RANK.get(occ_c, 3) < current_rank
                if r["adjusted"] < target and better_cat:
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

@st.cache_data
def compute_daily_curve(linea, estacion, fecha_str, _historical_sample):
    fecha    = datetime.date.fromisoformat(fecha_str)
    dow      = fecha.weekday()
    slots    = [
        t for t in [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]
        if _OPENING[dow] <= t <= _CLOSING.get(linea, {}).get(dow, "23:00")
    ]
    records = []
    for t in slots:
        h, m = int(t[:2]), int(t[3:])
        ts   = datetime.datetime(fecha.year, fecha.month, fecha.day, h, m)
        try:
            r = predictor.predict(linea, estacion, ts, _historical_sample)
            records.append({"hora": t, "carga": max(0.0, r["adjusted"])})
        except Exception:
            records.append({"hora": t, "carga": 0.0})
    return pd.DataFrame(records)


model, meta = init_model()
predictor._MODEL = model
predictor._META  = meta
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
    event    = res["event_applied"]

    occ_text, _ = occupancy_label(adjusted)

    st.markdown(f"#### {linea} · {estacion}")
    st.markdown(f"# {occ_text}")

    if event is not None:
        delta_pct = (adjusted - base) / max(base, 1) * 100
        st.caption(f"Ajuste por evento: {delta_pct:+.1f}%")
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
            f"vas a viajar mejor: {occ_better}"
        )
else:
    st.info("Seleccioná los parámetros y presioná **Predecir** para ver el resultado.")

# ---------------------------------------------------------------------------
# Panel principal — Sección 2: curva diaria
# ---------------------------------------------------------------------------
st.divider()
st.subheader(f"Ocupación a lo largo del día — {estacion}")

mask_chart = (
    (historical_df["LINEA"] == linea) &
    (historical_df["ESTACION"] == estacion)
)
with st.spinner("Calculando curva del día..."):
    df_curve = compute_daily_curve(linea, estacion, str(fecha), historical_df[mask_chart].copy())

if not df_curve.empty:
    max_y = max(df_curve["carga"].max() * 1.15, 150)
    fig   = go.Figure()

    fig.add_hrect(y0=0,   y1=40,    fillcolor="#22c55e", opacity=0.08, line_width=0)
    fig.add_hrect(y0=40,  y1=80,    fillcolor="#eab308", opacity=0.08, line_width=0)
    fig.add_hrect(y0=80,  y1=130,   fillcolor="#f97316", opacity=0.08, line_width=0)
    fig.add_hrect(y0=130, y1=max_y, fillcolor="#ef4444", opacity=0.08, line_width=0)

    fig.add_trace(go.Scatter(
        x=df_curve["hora"], y=df_curve["carga"],
        mode="lines", fill="tozeroy",
        line=dict(color="#2DE1C2", width=2),
        fillcolor="rgba(45, 225, 194, 0.12)",
        hovertemplate="%{x} — %{y:.0f} pax<extra></extra>",
    ))

    fig.add_vline(
        x=hora_str,
        line_dash="dash", line_color="rgba(255,255,255,0.5)", line_width=1.5,
    )

    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Panel principal — Sección 3: eventos activos
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
