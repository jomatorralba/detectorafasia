"""
app_disartria.py — Evaluación clínica de disartria para logopedas
Pruebas: TMF · Diadococinesias · Lectura del Abuelo · Lectura de Palabras
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import librosa

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluación de Disartria",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { max-width: 960px; padding-top: 1.5rem; }

  .word-card {
    text-align: center; padding: 32px 20px;
    background: linear-gradient(135deg, #1a237e, #3949ab);
    border-radius: 16px; color: white; margin: 18px 0;
  }
  .word-card .word { font-size: 3.2rem; font-weight: 900; letter-spacing: 6px; }
  .word-card .meta { opacity: .75; font-size: .9rem; margin-top: 6px; }

  .instr {
    background: #e3f2fd; border-left: 4px solid #1976d2;
    padding: 14px 18px; border-radius: 0 8px 8px 0; margin: 12px 0 18px 0;
  }
  .instr b { color: #0d47a1; }

  .badge        { text-align: center; border-radius: 10px; padding: 12px; margin-top: 10px; }
  .badge-green  { background: #e8f5e9; border: 1.5px solid #43a047; }
  .badge-yellow { background: #fff8e1; border: 1.5px solid #f9a825; }
  .badge-red    { background: #ffebee; border: 1.5px solid #e53935; }
  .badge .val   { font-size: 1.6rem; font-weight: bold; }
  .badge .lbl   { font-size: .85rem; color: #555; }
  .badge .ref   { font-size: .78rem; color: #888; margin-top: 2px; }

  .text-box {
    background: #fafafa; border: 1px solid #e0e0e0; border-radius: 10px;
    padding: 22px 26px; font-size: 1.2rem; line-height: 2.0;
    font-family: Georgia, serif; color: #212121; margin: 14px 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Datos normativos ───────────────────────────────────────────────────────────
NORM = {
    "tmf_a":  dict(min=10.0, display_max=35.0),
    "tmf_s":  dict(min=8.0,  display_max=35.0),
    "ratio":  dict(min=0.70, max=1.40),
    "ddk":    dict(min=5.0,  display_max=12.0),
    "wpm":    dict(min=100,  display_max=200),
}

# Texto "El Abuelo" (lectura clínica estandarizada)
ABUELO = (
    "Abuelo viene a visitarnos todos los domingos. "
    "Siempre trae caramelos para los niños y flores para la abuela. "
    "Le gusta sentarse en el jardín y contarnos historias de cuando era joven. "
    "Dice que antes la vida era más sencilla pero también más dura. "
    "Trabajaba en el campo desde muy temprano hasta que se ponía el sol. "
    "Ahora descansa y cuida su huerto con mucho cariño. "
    "Los tomates de su huerto son los más ricos que he probado nunca. "
    "Cuando nos vamos a casa siempre nos da un abrazo muy fuerte y cariñoso."
)
ABUELO_N = len(ABUELO.split())

# Palabras test con duración de referencia en hablante sano (segundos)
PALABRAS = {
    "PIANO":         dict(syl=2, ref=0.65),
    "LAICO":         dict(syl=3, ref=0.75),
    "CALENDARIO":    dict(syl=5, ref=1.10),
    "AUTOBIOGRAFÍA": dict(syl=7, ref=1.55),
}

# ── Estado de sesión ───────────────────────────────────────────────────────────
if "res" not in st.session_state:
    st.session_state.res = {}

# ── Utilidades de audio ────────────────────────────────────────────────────────
def load_audio(f, sr=22050):
    """Carga audio desde st.audio_input (BytesIO/UploadedFile)."""
    y, _ = librosa.load(io.BytesIO(f.read()), sr=sr, mono=True)
    return y, sr


def calc_tmf(y, sr):
    """Duración de fonación usando umbral RMS relativo."""
    fl = int(0.025 * sr)
    hl = int(0.010 * sr)
    n  = max(1, (len(y) - fl) // hl + 1)
    rms = np.array([
        np.sqrt(np.mean(y[i * hl: i * hl + fl] ** 2)) for i in range(n)
    ])
    if rms.max() == 0:
        return 0.0
    idx = np.where(rms > rms.max() * 0.05)[0]
    if not len(idx):
        return 0.0
    return float((idx[-1] - idx[0]) * hl / sr)


def calc_ddk(y, sr):
    """Estima repeticiones de PATAKA por detección de onsets silábicos."""
    onset_f = librosa.onset.onset_detect(
        y=y, sr=sr, units="frames",
        delta=0.35, wait=8, pre_max=3, post_max=3, pre_avg=3, post_avg=5,
    )
    onset_t = librosa.frames_to_time(onset_f, sr=sr, hop_length=512)
    return len(onset_t) / 3.0, onset_t


# ── Helpers de gráficas ────────────────────────────────────────────────────────
C = dict(blue="#1565C0", green="#2e7d32", purple="#6a1b9a")


def _layout(title, h):
    return dict(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Tiempo (s)", height=h,
        margin=dict(t=38, b=35, l=42, r=16),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
    )


def waveform_fig(y, sr, title="", color="blue", h=190):
    step = max(1, len(y) // 3000)
    t = np.linspace(0, len(y) / sr, len(y))[::step]
    fig = go.Figure(go.Scatter(
        x=t, y=y[::step], mode="lines",
        line=dict(color=C[color], width=1),
    ))
    fig.update_layout(**_layout(title, h))
    fig.update_yaxes(range=[-1, 1])
    return fig


def gauge_fig(val, norm_min, dmax, title, unit):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number=dict(suffix=f" {unit}", font=dict(size=22)),
        title=dict(text=title, font=dict(size=13)),
        gauge=dict(
            axis=dict(range=[0, dmax]),
            bar=dict(color=C["blue"], thickness=0.25),
            steps=[
                dict(range=[0, norm_min * 0.7],  color="#ffcdd2"),
                dict(range=[norm_min * 0.7, norm_min], color="#fff9c4"),
                dict(range=[norm_min, dmax],      color="#c8e6c9"),
            ],
            threshold=dict(
                line=dict(color="#b71c1c", width=3),
                thickness=0.8, value=norm_min,
            ),
        ),
    ))
    fig.update_layout(height=175, margin=dict(t=38, b=8, l=8, r=8))
    return fig


def word_fig(y, sr, word):
    """Envolvente de amplitud grabada vs referencia normalizada."""
    ref = PALABRAS[word]["ref"]
    dur = len(y) / sr
    y_n = y / (np.max(np.abs(y)) + 1e-8)

    n = 200
    chunk = max(1, len(y_n) // n)
    env   = np.array([np.max(np.abs(y_n[i * chunk:(i + 1) * chunk])) for i in range(n)])
    env_t = np.linspace(0, dur, n)

    # Envolvente gaussiana de referencia (hablante sano)
    t_ref = np.linspace(0, max(dur * 1.15, ref * 1.4), 300)
    mu, sig = ref / 2, ref / 4
    ref_c = np.exp(-0.5 * ((t_ref - mu) / sig) ** 2) * 0.85

    fig = go.Figure()
    # Banda de normalidad (sombreada)
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_ref, t_ref[::-1]]),
        y=np.concatenate([ref_c * 1.25, (ref_c * 0.75)[::-1]]),
        fill="toself", fillcolor="rgba(46,125,50,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Rango normal", hoverinfo="skip",
    ))
    # Línea media de referencia
    fig.add_trace(go.Scatter(
        x=t_ref, y=ref_c, mode="lines",
        line=dict(color="rgba(46,125,50,.65)", width=2, dash="dash"),
        name=f"Referencia (~{ref} s)",
    ))
    # Envolvente grabada
    fig.add_trace(go.Scatter(
        x=env_t, y=env, mode="lines",
        line=dict(color=C["blue"], width=2.5),
        name="Grabación",
    ))
    fig.add_vline(
        x=ref, line_dash="dot", line_color="rgba(46,125,50,.7)",
        annotation_text=f" ref {ref} s", annotation_font_size=11,
    )
    lo = _layout(f"Envolvente: {word}", 255)
    lo["yaxis_title"] = "Amplitud norm."
    lo["legend"] = dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=11))
    fig.update_layout(**lo)
    return fig, dur


# ── Badge de resultado ─────────────────────────────────────────────────────────
def classify(val, key, mode="min"):
    n = NORM[key]
    if mode == "min":
        ok   = val >= n["min"]
        warn = val >= n["min"] * 0.7
    else:
        ok   = n["min"] <= val <= n["max"]
        warn = n["min"] * 0.7 <= val <= n["max"] * 1.3
    if ok:   return "badge-green",  "🟢", "Normal"
    if warn: return "badge-yellow", "🟡", "Límite"
    return        "badge-red",   "🔴", "Alterado"


def show_badge(label, val_str, cls, icon, text, ref_str):
    st.markdown(f"""
    <div class="badge {cls}">
      <div class="val">{icon} {val_str}</div>
      <div class="lbl">{label} · <b>{text}</b></div>
      <div class="ref">Referencia: {ref_str}</div>
    </div>""", unsafe_allow_html=True)


# ── Prueba 1 · TMF ─────────────────────────────────────────────────────────────
def tab_tmf():
    st.header("⏱️ Tiempo Máximo de Fonación")
    st.markdown("""<div class="instr"><b>Instrucciones para el paciente</b><br>
    Tome aire profundamente. Emita el sonido indicado de forma sostenida y uniforme
    lo más tiempo posible. Pare cuando se le acabe el aire.</div>""",
    unsafe_allow_html=True)

    ca, cs = st.columns(2)

    with ca:
        st.subheader("Vocal /A/")
        st.caption("Emitir **A** sostenida lo máximo posible")
        af = st.audio_input("▶ Grabar vocal /A/", key="tmf_a")
        if af:
            with st.spinner("Procesando…"):
                y, sr = load_audio(af)
                d = calc_tmf(y, sr)
            st.session_state.res["tmf_a"] = d
            st.plotly_chart(waveform_fig(y, sr, f"A — {d:.1f} s"), use_container_width=True)
            st.plotly_chart(
                gauge_fig(d, NORM["tmf_a"]["min"], NORM["tmf_a"]["display_max"], "TMF /A/", "s"),
                use_container_width=True,
            )
            show_badge("TMF /A/", f"{d:.1f} s",
                       *classify(d, "tmf_a"), f"≥ {NORM['tmf_a']['min']} s")

    with cs:
        st.subheader("Fricativa /S/")
        st.caption("Emitir **S** sostenida lo máximo posible")
        sf_f = st.audio_input("▶ Grabar fricativa /S/", key="tmf_s")
        if sf_f:
            with st.spinner("Procesando…"):
                y, sr = load_audio(sf_f)
                d = calc_tmf(y, sr)
            st.session_state.res["tmf_s"] = d
            st.plotly_chart(waveform_fig(y, sr, f"S — {d:.1f} s", "purple"), use_container_width=True)
            st.plotly_chart(
                gauge_fig(d, NORM["tmf_s"]["min"], NORM["tmf_s"]["display_max"], "TMF /S/", "s"),
                use_container_width=True,
            )
            show_badge("TMF /S/", f"{d:.1f} s",
                       *classify(d, "tmf_s"), f"≥ {NORM['tmf_s']['min']} s")

    r = st.session_state.res
    if "tmf_a" in r and "tmf_s" in r and r["tmf_a"] > 0:
        ratio = r["tmf_s"] / r["tmf_a"]
        r["ratio_sa"] = ratio
        st.divider()
        st.subheader("📐 Cociente S/A")
        c1, c2, c3 = st.columns(3)
        c1.metric("TMF /A/", f"{r['tmf_a']:.1f} s")
        c2.metric("TMF /S/", f"{r['tmf_s']:.1f} s")
        with c3:
            show_badge("Cociente S/A", f"{ratio:.2f}",
                       *classify(ratio, "ratio", "range"),
                       f"{NORM['ratio']['min']}–{NORM['ratio']['max']}")
        if ratio > 1.4:
            st.info("💡 Cociente > 1.4 → posible patología glótica.")
        elif ratio < 0.7:
            st.info("💡 Cociente < 0.7 → posible patología respiratoria o insuficiencia velofaríngea.")


# ── Prueba 2 · Diadococinesias ─────────────────────────────────────────────────
def tab_ddk():
    st.header("🗣️ Diadococinesias")
    st.markdown("""<div class="instr"><b>Instrucciones para el paciente</b><br>
    Tome aire y repita <b>PA·TA·KA</b> tan rápido y claro como pueda durante
    <b>5 segundos</b>. Pulse stop al terminar.</div>""",
    unsafe_allow_html=True)

    af = st.audio_input("▶ Grabar PATAKA (~5 segundos)", key="ddk")
    if af:
        with st.spinner("Contando sílabas…"):
            y, sr = load_audio(af)
            dur = len(y) / sr
            n_pat, onsets = calc_ddk(y, sr)

        dur_eval = min(dur, 5.0)
        st.session_state.res["ddk"] = n_pat

        # Forma de onda con marcadores de onset
        step = max(1, len(y) // 3000)
        t = np.linspace(0, dur, len(y))[::step]
        fig = go.Figure(go.Scatter(x=t, y=y[::step], mode="lines",
                                   line=dict(color=C["blue"], width=1)))
        for ot in onsets:
            if ot <= dur:
                fig.add_vline(x=ot, line_color="rgba(239,83,80,.5)", line_width=1.5)
        if dur > 5:
            fig.add_vline(x=5, line_color=C["green"], line_dash="dash",
                          annotation_text=" 5 s", annotation_font_size=11)
        fig.update_layout(**_layout(
            f"Forma de onda — {len(onsets)} sílabas detectadas → {n_pat:.1f} PATAKA", 215))
        fig.update_yaxes(range=[-1, 1])
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Duración evaluada", f"{dur_eval:.1f} s")
        c2.metric("PATAKA detectados", f"{n_pat:.1f}")
        c3.metric("Tasa", f"{n_pat / dur_eval:.2f} /s" if dur_eval else "—")

        show_badge("Diadococinesias", f"{n_pat:.1f} rep/5s",
                   *classify(n_pat, "ddk"), f"≥ {NORM['ddk']['min']} rep/5s")

        st.divider()
        st.caption("¿El conteo automático no es preciso? Corrígelo aquí:")
        manual = st.number_input("Repeticiones PATAKA (manual)", 0, 50,
                                  value=round(n_pat), key="ddk_manual")
        if manual != round(n_pat):
            st.session_state.res["ddk"] = float(manual)
            show_badge("Resultado corregido", f"{manual} rep/5s",
                       *classify(float(manual), "ddk"), f"≥ {NORM['ddk']['min']} rep/5s")


# ── Prueba 3 · Lectura del Abuelo ─────────────────────────────────────────────
def tab_abuelo():
    st.header("📖 Lectura del Abuelo")
    st.markdown("""<div class="instr"><b>Instrucciones para el paciente</b><br>
    Lea el texto en voz alta a su ritmo habitual, de forma fluida y natural.
    Pulse stop cuando haya terminado.</div>""",
    unsafe_allow_html=True)

    st.markdown(
        f'<div class="text-box">{ABUELO}</div>'
        f'<p style="color:#999;font-size:.82rem;margin-top:-6px">📝 {ABUELO_N} palabras</p>',
        unsafe_allow_html=True,
    )

    af = st.audio_input("▶ Grabar lectura completa", key="abuelo")
    if af:
        with st.spinner("Calculando velocidad…"):
            y, sr = load_audio(af)
            dur = len(y) / sr
            wpm = (ABUELO_N / dur * 60) if dur > 0 else 0
        st.session_state.res["wpm"] = wpm

        st.plotly_chart(
            waveform_fig(y, sr, f"Lectura — {dur:.1f} s · {wpm:.0f} pal/min", "green", 200),
            use_container_width=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Duración total", f"{dur:.1f} s")
        c2.metric("Palabras / minuto", f"{wpm:.0f}")
        c3.metric("Mínimo normal", f"≥ {NORM['wpm']['min']} ppm")

        show_badge("Velocidad lectora", f"{wpm:.0f} ppm",
                   *classify(wpm, "wpm"), f"≥ {NORM['wpm']['min']} ppm")

        if wpm < 70:
            st.error("Velocidad muy reducida · posible disartria moderada-severa.")
        elif wpm < 100:
            st.warning("Velocidad por debajo del rango normal.")
        elif wpm < 130:
            st.warning("Velocidad ligeramente reducida o en límite inferior.")
        else:
            st.success("Velocidad dentro del rango normal.")


# ── Prueba 4 · Lectura de Palabras ────────────────────────────────────────────
def tab_palabras():
    st.header("🔤 Lectura de Palabras")
    st.markdown("""<div class="instr"><b>Instrucciones para el paciente</b><br>
    Lea la palabra que aparece en pantalla en voz alta, de forma clara y natural.
    Grabamos una palabra a la vez. Pulse stop al terminar la palabra.</div>""",
    unsafe_allow_html=True)

    if "palabras" not in st.session_state.res:
        st.session_state.res["palabras"] = {}

    word = st.radio("Selecciona la palabra:", list(PALABRAS.keys()),
                    horizontal=True, key="word_sel")
    ref = PALABRAS[word]

    st.markdown(f"""<div class="word-card">
      <div class="word">{word}</div>
      <div class="meta">{ref['syl']} sílabas · duración esperada ≈ {ref['ref']} s</div>
    </div>""", unsafe_allow_html=True)

    af = st.audio_input(f"▶ Grabar '{word}'", key=f"w_{word}")
    if af:
        with st.spinner("Analizando…"):
            y, sr = load_audio(af)
            fig, dur = word_fig(y, sr, word)
        st.session_state.res["palabras"][word] = dur
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Duración grabada", f"{dur:.2f} s")
        c2.metric("Referencia normal", f"~{ref['ref']} s")
        ratio = dur / ref["ref"]
        diff  = dur - ref["ref"]
        c3.metric("Diferencia", f"{diff:+.2f} s",
                  "Normal" if ratio <= 1.5 else "Lento",
                  delta_color="normal" if ratio <= 1.5 else "inverse")

    st.divider()
    done = len(st.session_state.res.get("palabras", {}))
    st.progress(done / len(PALABRAS), text=f"Palabras grabadas: {done}/{len(PALABRAS)}")

    if st.session_state.res.get("palabras"):
        rows = [
            dict(
                Palabra=w,
                Duración=f"{d:.2f} s",
                Referencia=f"{PALABRAS[w]['ref']} s",
                Ratio=f"{d / PALABRAS[w]['ref']:.2f}×",
                Estado="✅ Normal" if d / PALABRAS[w]["ref"] <= 1.5 else "⚠️ Lento",
            )
            for w, d in st.session_state.res["palabras"].items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ── Resumen ────────────────────────────────────────────────────────────────────
def tab_resumen():
    st.header("📊 Resumen de Evaluación")
    r = st.session_state.res

    if not r:
        st.info("Completa al menos una prueba para ver el resumen.")
        return

    NORM_KEY_MAP = {"ratio_sa": "ratio"}
    defs = [
        ("tmf_a",    "TMF /A/",             "min",   f"≥ {NORM['tmf_a']['min']} s",       lambda v: f"{v:.1f} s"),
        ("tmf_s",    "TMF /S/",             "min",   f"≥ {NORM['tmf_s']['min']} s",       lambda v: f"{v:.1f} s"),
        ("ratio_sa", "Cociente S/A",        "range", f"{NORM['ratio']['min']}–{NORM['ratio']['max']}", lambda v: f"{v:.2f}"),
        ("ddk",      "Diadococinesias",     "min",   f"≥ {NORM['ddk']['min']} rep/5s",    lambda v: f"{v:.1f}"),
        ("wpm",      "Velocidad lectora",   "min",   f"≥ {NORM['wpm']['min']} ppm",       lambda v: f"{v:.0f} ppm"),
    ]

    rows = []
    for key, label, mode, ref_str, fmt in defs:
        if key in r:
            nk = NORM_KEY_MAP.get(key, key)
            _, icon, text = classify(r[key], nk, mode)
            rows.append({"Prueba": label, "Resultado": fmt(r[key]),
                         "Referencia": ref_str, "Estado": f"{icon} {text}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if r.get("palabras"):
        st.subheader("Lectura de palabras")
        wrows = [
            dict(
                Palabra=w,
                Duración=f"{d:.2f} s",
                Referencia=f"{PALABRAS[w]['ref']} s",
                Ratio=f"{d / PALABRAS[w]['ref']:.2f}×",
                Estado="✅ Normal" if d / PALABRAS[w]["ref"] <= 1.5 else "⚠️ Lento",
            )
            for w, d in r["palabras"].items()
        ]
        st.dataframe(pd.DataFrame(wrows), hide_index=True, use_container_width=True)

    st.divider()
    if st.button("🔄 Nueva evaluación", type="secondary"):
        st.session_state.res = {}
        st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div style="padding:4px 0 18px 0">
      <h1 style="margin:0;font-size:1.9rem">🎙️ Evaluación de Disartria</h1>
      <p style="color:#666;margin:4px 0 0 0;font-size:.9rem">
        Herramienta clínica para logopedas ·
        Esta herramienta no sustituye la valoración profesional
      </p>
    </div>""", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs([
        "⏱️ TMF",
        "🗣️ Diadococinesias",
        "📖 Lectura del Abuelo",
        "🔤 Lectura de Palabras",
        "📊 Resumen",
    ])
    with t1: tab_tmf()
    with t2: tab_ddk()
    with t3: tab_abuelo()
    with t4: tab_palabras()
    with t5: tab_resumen()


if __name__ == "__main__":
    main()
