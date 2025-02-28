import os
import tempfile
from datetime import datetime
import numpy as np
import streamlit as st
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import spacy
from spacy.lang.es.examples import sentences
import difflib
from metaphone import doublemetaphone
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write

# Cargar modelos de NLP y reconocimiento de voz (se cargarán la primera vez)
@st.cache_resource
def load_models():
    # Modelo de reconocimiento de voz (español)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")
    
    # Modelo de NLP para análisis semántico
    nlp = spacy.load("es_core_news_md")
    
    return processor, model, nlp

# Preprocesar audio para el reconocimiento
def process_audio(audio_file, processor, model):
    # Cargar y normalizar audio
    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    
    # Procesar con Wav2Vec2
    with torch.no_grad():
        input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription.lower()

# Analizar tipos de errores
def analizar_respuesta(respuesta, palabra_objetivo, nlp):
    # Limpiar y normalizar las palabras
    respuesta = respuesta.strip().lower()
    palabra_objetivo = palabra_objetivo.strip().lower()
    
    # Si la respuesta es correcta
    if respuesta == palabra_objetivo:
        return "Respuesta correcta", 1.0, 0.0, 0.0, "✅"
    
    # Análisis fonológico
    ratio_fonologico = difflib.SequenceMatcher(None, respuesta, palabra_objetivo).ratio()
    
    # Análisis fonético con metaphone
    metafona_respuesta = doublemetaphone(respuesta)[0]
    metafona_objetivo = doublemetaphone(palabra_objetivo)[0]
    similitud_fonetica = difflib.SequenceMatcher(None, metafona_respuesta, metafona_objetivo).ratio()
    
    # Análisis semántico
    doc_respuesta = nlp(respuesta)
    doc_objetivo = nlp(palabra_objetivo)
    
    # Verificar que ambas palabras estén en el vocabulario
    if doc_respuesta.vector_norm and doc_objetivo.vector_norm:
        similitud_semantica = doc_respuesta.similarity(doc_objetivo)
    else:
        similitud_semantica = 0.0
    
    # Clasificar el tipo de error
    if ratio_fonologico > 0.7 or similitud_fonetica > 0.8:
        tipo_error = "Parafasia fonológica"
        emoji = "🔵"
    elif similitud_semantica > 0.5:
        tipo_error = "Parafasia semántica"
        emoji = "🟢" 
    elif respuesta == "":
        tipo_error = "Sin respuesta"
        emoji = "⚪"
    elif len(respuesta.split()) > 3:  
        tipo_error = "Circunloquio"
        emoji = "🟠"
    else:
        tipo_error = "Neologismo o error no clasificado"
        emoji = "🔴"
    
    return tipo_error, ratio_fonologico, similitud_fonetica, similitud_semantica, emoji

# Grabar audio usando sounddevice
def grabar_audio(duracion=5, fs=16000):
    st.write("🎙️ Grabando...")
    myrecording = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
    sd.wait()
    st.write("✅ Grabación completada!")
    
    # Crear archivo temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    write(temp_file.name, fs, myrecording)
    
    return temp_file.name

# Interfaz de la aplicación con Streamlit
def main():
    st.set_page_config(page_title="Análisis de Parafasias", page_icon="🗣️")
    
    # Cargar modelos
    with st.spinner('Cargando modelos de IA...'):
        processor, model, nlp = load_models()
    
    # Título y descripción
    st.title("🗣️ Detector de Parafasias")
    st.write("""
    Esta aplicación analiza la voz del usuario para detectar parafasias y otros errores 
    en ejercicios de denominación de palabras. Útil para terapia del lenguaje 
    y evaluación de trastornos del habla.
    """)
    
    # Sidebar para configuración
    st.sidebar.title("⚙️ Configuración")
    
    # Lista predefinida de palabras para denominar
    palabras_predefinidas = {
        "Básico": ["casa", "perro", "lápiz", "mesa", "libro"],
        "Objetos comunes": ["teléfono", "cuchara", "zapato", "reloj", "puerta"],
        "Animales": ["elefante", "jirafa", "cocodrilo", "mariposa", "canguro"],
        "Alimentos": ["manzana", "espagueti", "zanahoria", "hamburguesa", "plátano"],
        "Transportes": ["helicóptero", "motocicleta", "submarino", "ambulancia", "bicicleta"]
    }
    
    # Seleccionar lista o palabra personalizada
    modo = st.sidebar.radio("Modo de trabajo", ["Listas predefinidas", "Palabra personalizada"])
    
    if modo == "Listas predefinidas":
        categoria = st.sidebar.selectbox("Categoría", list(palabras_predefinidas.keys()))
        palabra_objetivo = st.sidebar.selectbox("Palabra a denominar", palabras_predefinidas[categoria])
    else:
        palabra_objetivo = st.sidebar.text_input("Palabra a denominar", "casa")
    
    # Configuración de grabación
    duracion_grabacion = st.sidebar.slider("Duración de grabación (segundos)", 2, 10, 5)
    
    # Visualización de historial
    if 'historial' not in st.session_state:
        st.session_state.historial = []
    
    # Sección principal
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader(f"Palabra a denominar: {palabra_objetivo.upper()}")
        st.write("Presiona el botón para grabar tu voz pronunciando la palabra.")
        
        if st.button("🎙️ Grabar Voz"):
            # Grabar audio
            audio_file = grabar_audio(duracion=duracion_grabacion)
            
            # Reproducir grabación
            st.audio(audio_file, format="audio/wav")
            
            # Reconocer y analizar
            with st.spinner('Procesando audio...'):
                # Reconocimiento de voz
                transcripcion = process_audio(audio_file, processor, model)
                st.write(f"**Texto reconocido:** {transcripcion}")
                
                # Analizar errores
                tipo_error, ratio_fono, ratio_meta, similitud_sem, emoji = analizar_respuesta(
                    transcripcion, palabra_objetivo, nlp
                )
                
                # Mostrar resultados
                st.write(f"**Resultado:** {emoji} {tipo_error}")
                
                # Guardar en historial
                st.session_state.historial.append({
                    "fecha": datetime.now().strftime("%H:%M:%S"),
                    "palabra_objetivo": palabra_objetivo,
                    "respuesta": transcripcion,
                    "tipo_error": tipo_error,
                    "similitud_fonologica": round(ratio_fono, 2),
                    "similitud_fonetica": round(ratio_meta, 2),
                    "similitud_semantica": round(similitud_sem, 2)
                })
                
                # Limpiar archivo temporal
                os.unlink(audio_file)
        
        # Mostrar historial de intentos
        if st.session_state.historial:
            st.subheader("Historial de Intentos")
            df = pd.DataFrame(st.session_state.historial)
            st.dataframe(df)
            
            if st.button("Limpiar historial"):
                st.session_state.historial = []
                st.experimental_rerun()
    
    with col2:
        if st.session_state.historial:
            st.subheader("Análisis")
            
            # Crear gráficos de rendimiento
            fig, ax = plt.subplots(figsize=(5, 4))
            
            # Contar tipos de errores
            tipos_errores = [h["tipo_error"] for h in st.session_state.historial]
            conteo = pd.Series(tipos_errores).value_counts()
            
            # Gráfico de barras
            conteo.plot(kind='barh', ax=ax, color=['green', 'blue', 'orange', 'red', 'gray'])
            ax.set_title("Tipos de errores")
            st.pyplot(fig)
            
            # Gráfico de similitud a lo largo del tiempo
            if len(st.session_state.historial) > 1:
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                df_hist = pd.DataFrame(st.session_state.historial)
                df_hist[['similitud_fonologica', 'similitud_semantica']].plot(ax=ax2)
                ax2.set_title("Evolución de similitud")
                ax2.set_ylim(0, 1)
                st.pyplot(fig2)
    
    # Información adicional
    st.subheader("📋 Tipos de Errores")
    st.markdown("""
    - ✅ **Respuesta correcta**: Coincide exactamente con la palabra objetivo.
    - 🔵 **Parafasia fonológica**: Error en los sonidos pero mantiene estructura similar.
    - 🟢 **Parafasia semántica**: Palabra relacionada semánticamente con el objetivo.
    - 🟠 **Circunloquio**: Descripción en lugar de denominación directa.
    - 🔴 **Neologismo/Otro**: Palabra inventada o error no clasificable.
    - ⚪ **Sin respuesta**: No se detectó respuesta verbal.
    """)
    
    st.markdown("---")
    st.caption("Desarrollado para evaluación y terapia de trastornos del lenguaje. Esta herramienta no sustituye la evaluación profesional.")

if __name__ == "__main__":
    main()