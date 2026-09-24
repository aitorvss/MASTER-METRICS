import warnings
warnings.filterwarnings("ignore")

import os
import json
import re

import streamlit as st
import librosa
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="MasterMetrics AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CSS
# ============================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    animation: fadeIn 0.8s ease-out;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

[data-testid="stFileUploadDropzone"] {
    min-height: 200px !important;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #16161a !important;
    border: 2px dashed #4b4b55 !important;
    border-radius: 15px !important;
    transition: all 0.3s ease;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: #1DB954 !important;
    background-color: #1e1e24 !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > div {
    border-radius: 8px !important;
    border: 1px solid #4b4b55 !important;
    background-color: #1e1e24 !important;
}

.stSelectbox div[data-baseweb="select"] {
    white-space: normal !important;
}

[data-testid="stMetric"] {
    background-color: #1e1e24;
    border: 1px solid #2d2d35;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 15px rgba(29, 185, 84, 0.1);
    border-color: #1DB954;
}

[data-testid="stMetricValue"] > div {
    overflow: visible !important;
    white-space: nowrap !important;
}

.alerta-clipping {
    color: #ff4b4b;
    font-size: 1.8rem;
    font-weight: bold;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% {
        transform: scale(1);
    }

    50% {
        transform: scale(1.05);
        text-shadow: 0 0 10px rgba(255, 75, 75, 0.8);
    }

    100% {
        transform: scale(1);
    }
}

.alerta-segura {
    color: #1DB954;
    font-size: 1.8rem;
    font-weight: bold;
}

.metric-label {
    font-size: 14px;
    color: rgba(250, 250, 250, 0.6);
    margin-bottom: 5px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. GEMINI
# ============================================================
#
# IMPORTANTE:
# NO pongas la API key directamente en el código.
#
# Para Streamlit:
#
# .streamlit/secrets.toml
#
# GEMINI_API_KEY="TU_CLAVE"
#
# ============================================================

GEMINI_API_KEY = ""

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# Intentamos usar el SDK nuevo de Google.
# Si no está instalado, utilizamos el antiguo como fallback.

USE_NEW_GEMINI = False
gemini_client = None
legacy_genai = None

if GEMINI_API_KEY:

    try:
        from google import genai

        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        USE_NEW_GEMINI = True

    except ImportError:

        try:
            import google.generativeai as legacy_genai

            legacy_genai.configure(api_key=GEMINI_API_KEY)

        except ImportError:
            pass


# Modelo actual.
GEMINI_MODEL = "gemini-3.8-flash"


# ============================================================
# 4. CLASE PDF
# ============================================================

class GeneradorPDF(FPDF):

    def header(self):
        self.set_fill_color(22, 22, 26)
        self.rect(0, 0, 210, 297, "F")

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", size=9)
        self.set_text_color(100, 100, 100)
        self.cell(
            0,
            10,
            "Generado por MasterMetrics AI v1.3",
            align="C"
        )


# ============================================================
# 5. GENERADOR PDF
# ============================================================

def generar_pdf(informe, datos):

    pdf = GeneradorPDF()

    pdf.set_auto_page_break(
        auto=True,
        margin=20
    )

    pdf.add_page()

    # Título
    pdf.set_font(
        "Helvetica",
        style="B",
        size=20
    )

    pdf.set_text_color(
        255,
        255,
        255
    )

    pdf.cell(
        0,
        15,
        "MasterMetrics AI - Diagnostico Tecnico",
        ln=True,
        align="C"
    )

    # Caja de métricas
    pdf.set_y(30)

    pdf.set_fill_color(
        30,
        30,
        36
    )

    pdf.rect(
        10,
        30,
        190,
        15,
        "F"
    )

    pdf.set_y(34)

    pdf.set_font(
        "Helvetica",
        style="B",
        size=10
    )

    pdf.set_text_color(
        200,
        200,
        200
    )

    pdf.cell(
        38,
        7,
        f"Tempo: {datos['bpm']} BPM",
        align="C"
    )

    pdf.cell(
        38,
        7,
        f"Key: {datos['tonalidad']}",
        align="C"
    )

    pdf.cell(
        38,
        7,
        f"LUFS: {datos['lufs_integrados']}",
        align="C"
    )

    if datos["true_peak"] > 0:
        pdf.set_text_color(
            255,
            75,
            75
        )
    else:
        pdf.set_text_color(
            29,
            185,
            84
        )

    pdf.cell(
        38,
        7,
        f"Peak: {datos['true_peak']} dB",
        align="C"
    )

    pdf.set_text_color(
        200,
        200,
        200
    )

    pdf.cell(
        38,
        7,
        f"Crest: {datos['crest_factor']} dB",
        ln=True,
        align="C"
    )

    pdf.set_draw_color(
        75,
        75,
        85
    )

    pdf.line(
        10,
        52,
        200,
        52
    )

    # Informe
    pdf.set_y(60)

    texto_limpio = re.sub(
        r"[*_#]",
        "",
        informe
    )

    texto_limpio = (
        texto_limpio
        .encode("latin-1", "replace")
        .decode("latin-1")
    )

    pdf.set_font(
        "Helvetica",
        size=11
    )

    pdf.set_text_color(
        220,
        220,
        220
    )

    pdf.multi_cell(
        0,
        8,
        texto_limpio
    )

    return bytes(pdf.output())


# ============================================================
# 6. DETECTOR DE BPM
# ============================================================

def normalizar_bpm(bpm):
    """
    Corrige errores típicos de tempo:

    45  -> 90
    64  -> 128
    176 -> 88
    200 -> 100

    Mantiene BPM normales sin tocar.
    """

    bpm = float(bpm)

    if bpm <= 0:
        return 0

    # Subimos BPM demasiado bajos
    while bpm < 70:
        bpm *= 2

    # Bajamos BPM demasiado altos
    while bpm > 180:
        bpm /= 2

    return bpm


def estimar_bpm(y, sr):
    """
    Detector de BPM más robusto.

    1. Extrae la parte percusiva mediante HPSS.
    2. Calcula onset strength.
    3. Calcula tempo con feature.tempo.
    4. Calcula tempo con beat_track.
    5. Hace varias estimaciones.
    6. Normaliza errores de mitad/doble BPM.
    7. Escoge una estimación consensuada.
    """

    # --------------------------------------------------------
    # PARTE PERCUSIVA
    # --------------------------------------------------------

    try:
        _, y_percussive = librosa.effects.hpss(y)
    except Exception:
        y_percussive = y

    # --------------------------------------------------------
    # ONSET ENVELOPE
    # --------------------------------------------------------

    hop_length = 512

    onset_env = librosa.onset.onset_strength(
        y=y_percussive,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median
    )

    # Seguridad
    if len(onset_env) == 0:
        return 0

    # --------------------------------------------------------
    # ESTIMACIÓN 1: FEATURE TEMPO
    # --------------------------------------------------------

    candidatos = []

    try:

        tempo_feature = librosa.feature.tempo(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            start_bpm=120,
            max_tempo=200,
            aggregate=np.median
        )

        tempo_feature = float(
            np.atleast_1d(tempo_feature)[0]
        )

        if tempo_feature > 0:
            candidatos.append(tempo_feature)

    except Exception:
        pass

    # --------------------------------------------------------
    # ESTIMACIONES 2+: BEAT TRACK
    # --------------------------------------------------------

    start_bpms = [
        80,
        100,
        120,
        140,
        160
    ]

    for start_bpm in start_bpms:

        try:

            tempo, _ = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=hop_length,
                start_bpm=start_bpm,
                tightness=100,
                trim=True
            )

            tempo = float(
                np.atleast_1d(tempo)[0]
            )

            if tempo > 0:
                candidatos.append(tempo)

        except Exception:
            continue

    # --------------------------------------------------------
    # SI NO SE HA DETECTADO NADA
    # --------------------------------------------------------

    if not candidatos:
        return 0

    # --------------------------------------------------------
    # NORMALIZAR BPM
    # --------------------------------------------------------

    candidatos_normalizados = [
        normalizar_bpm(bpm)
        for bpm in candidatos
        if bpm > 0
    ]

    candidatos_normalizados = [
        bpm
        for bpm in candidatos_normalizados
        if 60 <= bpm <= 180
    ]

    if not candidatos_normalizados:
        return 0

    # --------------------------------------------------------
    # BUSCAR CONSENSO ENTRE ESTIMACIONES
    # --------------------------------------------------------

    # Agrupamos BPM cercanos.
    clusters = []

    for bpm in candidatos_normalizados:

        añadido = False

        for cluster in clusters:

            media_cluster = np.mean(cluster)

            if abs(bpm - media_cluster) <= 4:
                cluster.append(bpm)
                añadido = True
                break

        if not añadido:
            clusters.append([bpm])

    # Ordenamos por:
    # 1. cantidad de estimaciones
    # 2. cercanía entre ellas

    def score_cluster(cluster):

        cantidad = len(cluster)

        dispersion = np.std(cluster)

        return (
            cantidad * 10
            - dispersion
        )

    mejor_cluster = max(
        clusters,
        key=score_cluster
    )

    bpm_final = np.median(
        mejor_cluster
    )

    bpm_final = round(
        normalizar_bpm(bpm_final)
    )

    return int(bpm_final)


# ============================================================
# 7. ANÁLISIS COMPLETO DEL AUDIO
# ============================================================

def analizar_audio_completo(
    ruta_archivo,
    bpm_forzado
):

    # --------------------------------------------------------
    # INFORMACIÓN DEL ARCHIVO
    # --------------------------------------------------------

    info = sf.info(
        ruta_archivo
    )

    # Analizamos como máximo 180 segundos
    # para evitar consumir recursos innecesarios.

    frames_max = int(
        min(
            info.frames,
            info.samplerate * 180.0
        )
    )

    data, rate = sf.read(
        ruta_archivo,
        frames=frames_max
    )

    # Convertimos NaN/inf por seguridad
    data = np.nan_to_num(
        data
    )

    # --------------------------------------------------------
    # LOUDNESS / LUFS
    # --------------------------------------------------------

    meter = pyln.Meter(
        rate
    )

    try:

        lufs = round(
            meter.integrated_loudness(data),
            2
        )

    except Exception:

        # Si el archivo da algún problema,
        # intentamos mono.

        if data.ndim > 1:
            data_lufs = np.mean(
                data,
                axis=1
            )
        else:
            data_lufs = data

        lufs = round(
            meter.integrated_loudness(data_lufs),
            2
        )

    # --------------------------------------------------------
    # PEAK
    # --------------------------------------------------------

    peak_amplitude = np.max(
        np.abs(data)
    )

    peak = round(
        20 * np.log10(
            peak_amplitude + 1e-10
        ),
        2
    )

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms_amplitude = np.sqrt(
        np.mean(
            np.square(data)
        )
    )

    rms = (
        20 *
        np.log10(
            rms_amplitude + 1e-10
        )
    )

    # --------------------------------------------------------
    # CREST
    # --------------------------------------------------------

    crest = round(
        peak - rms,
        2
    )

    # --------------------------------------------------------
    # CARGAR AUDIO PARA BPM / KEY
    # --------------------------------------------------------

    # 22050 Hz es suficiente para tempo y análisis tonal
    # y reduce considerablemente el trabajo de CPU.

    y, sr = librosa.load(
        ruta_archivo,
        sr=22050,
        duration=180,
        mono=True
    )

    # Eliminamos silencios excesivos del principio/final.
    # Esto ayuda muchísimo cuando una canción tiene una intro
    # sin batería.

    try:

        y, _ = librosa.effects.trim(
            y,
            top_db=35
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # BPM
    # --------------------------------------------------------

    if bpm_forzado > 0:

        bpm_final = int(
            bpm_forzado
        )

    else:

        bpm_final = estimar_bpm(
            y,
            sr
        )

    # --------------------------------------------------------
    # TONALIDAD
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr
    )

    chroma_sum = np.sum(
        chroma,
        axis=1
    )

    # Perfiles de Krumhansl-Schmuckler
    maj_profile = [
        6.35,
        2.23,
        3.48,
        2.33,
        4.38,
        4.09,
        2.52,
        5.19,
        2.39,
        3.66,
        2.29,
        2.88
    ]

    min_profile = [
        6.33,
        2.68,
        3.52,
        5.38,
        2.60,
        3.53,
        2.54,
        4.75,
        3.98,
        2.69,
        3.34,
        3.17
    ]

    notas = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B"
    ]

    mejor_corr = -1
    mejor_tono = "No detectado"

    for i in range(12):

        # MAYOR
        corr_mayor = np.corrcoef(
            chroma_sum,
            np.roll(
                maj_profile,
                i
            )
        )[0, 1]

        if corr_mayor > mejor_corr:

            mejor_corr = corr_mayor

            mejor_tono = (
                f"{notas[i]} Mayor"
            )

        # MENOR
        corr_menor = np.corrcoef(
            chroma_sum,
            np.roll(
                min_profile,
                i
            )
        )[0, 1]

        if corr_menor > mejor_corr:

            mejor_corr = corr_menor

            mejor_tono = (
                f"{notas[i]} Menor"
            )

    # --------------------------------------------------------
    # FORMA DE ONDA
    # --------------------------------------------------------

    step = max(
        1,
        len(y) // 2500
    )

    y_plot = y[::step]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=y_plot,
            mode="lines",
            fill="tozeroy",
            line=dict(
                color="#1DB954",
                width=1
            )
        )
    )

    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),
        height=120,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_visible=False,
        yaxis_visible=False,
        hovermode=False
    )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    datos = {
        "bpm": bpm_final,
        "tonalidad": mejor_tono,
        "lufs_integrados": lufs,
        "true_peak": peak,
        "crest_factor": crest
    }

    return datos, fig


# ============================================================
# 8. GENERADOR DE INFORME IA
# ============================================================

def generar_informe_ia(
    datos,
    objetivo
):

    if not GEMINI_API_KEY:

        return (
            "No se ha configurado la API Key de Gemini.\n\n"
            "Configura GEMINI_API_KEY en Streamlit Secrets "
            "o como variable de entorno."
        )

    prompt = f"""
Eres un ingeniero de mezcla y mastering profesional.

Evalúa técnicamente estos datos de una canción
para el siguiente objetivo:

OBJETIVO:
{objetivo}

DATOS:
{json.dumps(datos, indent=4)}

Analiza:

1. LUFS:
   - Indica si el nivel es razonable para el objetivo.
   - Explica brevemente si debería modificarse.

2. Peak:
   - Avisa de problemas si está por encima de 0 dBFS.

3. Crest Factor:
   - Evalúa el rango dinámico.
   - Como referencia, aproximadamente 8-12 dB puede representar
     una dinámica razonable para muchos masters modernos.
   - Valores muy bajos pueden indicar un master muy comprimido.

4. BPM:
   - Simplemente indica el tempo detectado.

5. Tonalidad:
   - Indica la tonalidad detectada.

Sé conciso, técnico, estructurado y profesional.

No inventes datos que no estén presentes.

No utilices títulos con símbolos raros.
"""

    # --------------------------------------------------------
    # SDK NUEVO
    # --------------------------------------------------------

    if USE_NEW_GEMINI:

        try:

            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            return response.text

        except Exception as e:

            return (
                "Error al generar el informe con Gemini:\n\n"
                f"{str(e)}"
            )

    # --------------------------------------------------------
    # SDK ANTIGUO
    # --------------------------------------------------------

    if legacy_genai is not None:

        try:

            model = legacy_genai.GenerativeModel(
                GEMINI_MODEL
            )

            response = model.generate_content(
                prompt
            )

            return response.text

        except Exception as e:

            return (
                "Error al generar el informe con Gemini:\n\n"
                f"{str(e)}"
            )

    return (
        "No está instalado el SDK de Gemini.\n\n"
        "Instala google-genai."
    )


# ============================================================
# 9. MENÚ LATERAL
# ============================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3254/3254122.png",
        width=60
    )

    st.title(
        "Ajustes de Master"
    )

    st.markdown(
        "Configura tu objetivo acústico."
    )

    objetivo_master = st.selectbox(
        "Plataforma de destino:",
        options=[
            "Streaming (Spotify, Apple)",
            "Club (Alta energía)",
            "CD / Formato Físico",
            "Broadcast (TV/Radio)"
        ]
    )

    st.divider()

    st.caption(
        "MasterMetrics v1.3 - Motor Híbrido IA"
    )


# ============================================================
# 10. INTERFAZ PRINCIPAL
# ============================================================

st.title(
    "🎧 MasterMetrics AI Dashboard"
)

st.markdown(
    "Auditoría acústica profesional y diagnóstico por IA."
)


# ------------------------------------------------------------
# SUBIDA
# ------------------------------------------------------------

archivo_subido = st.file_uploader(
    "DropZone",
    type=["wav"],
    label_visibility="collapsed"
)


# ------------------------------------------------------------
# BPM MANUAL
# ------------------------------------------------------------

bpm_manual = st.number_input(
    "Corregir BPM (Opcional)",
    min_value=0,
    max_value=300,
    value=0,
    step=1,
    help=(
        "Déjalo en 0 para detección automática. "
        "Introduce un valor solo si quieres forzar el BPM."
    )
)

st.caption(
    "Déjalo en 0 para que MasterMetrics detecte automáticamente "
    "el BPM."
)


# ============================================================
# 11. ANÁLISIS
# ============================================================

if archivo_subido is not None:

    # --------------------------------------------------------
    # GUARDAR ARCHIVO TEMPORAL
    # --------------------------------------------------------

    temp_path = "temp.wav"

    try:

        with open(
            temp_path,
            "wb"
        ) as f:

            f.write(
                archivo_subido.getbuffer()
            )

        # ----------------------------------------------------
        # ANALIZAR
        # ----------------------------------------------------

        with st.status(
            "Analizando espectro y generando reporte...",
            expanded=True
        ) as status:

            try:

                datos, figura_onda = analizar_audio_completo(
                    temp_path,
                    bpm_manual
                )

                informe = generar_informe_ia(
                    datos,
                    objetivo_master
                )

                status.update(
                    label="Auditoría completada",
                    state="complete",
                    expanded=False
                )

                # --------------------------------------------
                # FORMA DE ONDA
                # --------------------------------------------

                st.divider()

                st.markdown(
                    "### Forma de Onda"
                )

                st.plotly_chart(
                    figura_onda,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    }
                )

                # --------------------------------------------
                # COLUMNAS
                # --------------------------------------------

                col_izq, col_der = st.columns(
                    [1, 1.8],
                    gap="large"
                )

                # --------------------------------------------
                # CUADRO DE MANDOS
                # --------------------------------------------

                with col_izq:

                    st.subheader(
                        "Cuadro de Mandos"
                    )

                    c1, c2 = st.columns(
                        2
                    )

                    c1.metric(
                        "Tempo",
                        f"{datos['bpm']} BPM"
                    )

                    c2.metric(
                        "Key",
                        datos["tonalidad"]
                    )

                    c1.metric(
                        "Loudness",
                        f"{datos['lufs_integrados']} LUFS"
                    )

                    with c2:

                        st.metric(
                            "True Peak",
                            f'{datos["true_peak"]} dB'
                        )

                        if datos["true_peak"] > 0:
                            st.error("⚠️ Clipping detectado")

                # --------------------------------------------
                # INFORME IA
                # --------------------------------------------

                with col_der:

                    st.subheader(
                        "🤖 Diagnóstico del Ingeniero"
                    )

                    with st.container(
                        height=420
                    ):

                        st.write(
                            informe
                        )

                    # ----------------------------------------
                    # PDF
                    # ----------------------------------------

                    pdf_bytes = generar_pdf(
                        informe,
                        datos
                    )

                    st.download_button(
                        label="📄 Descargar Informe Técnico (PDF)",
                        data=pdf_bytes,
                        file_name="MasterMetrics_Reporte.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )

            except Exception as e:

                error_text = str(e)

                if (
                    "429" in error_text
                    or "Quota exceeded" in error_text
                ):

                    st.warning(
                        "⏳ La API de Gemini está temporalmente "
                        "limitada. El análisis de audio sí puede "
                        "funcionar independientemente."
                    )

                else:

                    st.error(
                        f"Fallo técnico: {error_text}"
                    )

    finally:

        # ----------------------------------------------------
        # BORRAR TEMPORAL
        # ----------------------------------------------------

        if os.path.exists(
            temp_path
        ):

            try:
                os.remove(
                    temp_path
                )
            except Exception:
                pass