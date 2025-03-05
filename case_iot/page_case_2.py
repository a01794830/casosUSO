import streamlit as st
import logging
from case_iot.utils.pinecone_utils import query_by_id, interpret_and_search
from case_iot.utils.embedding_utils import generate_chat_response
from case_iot.utils.pdf_utils import generar_pdf

from config import Config

logger = logging.getLogger(__name__)

st.title("Caso 2 - Inteligencia Artificial Generativa IoT")

# Get project ID from environment variables
OPENAI_API_KEY = Config.OPENAI_API_KEY
PROJECT_ID = Config.BIGQUERY_PROJECT_ID

# CSS styles
st.markdown("""
<style>
.highlight {
    background-color: #eef8fa;
    padding: 15px;
    border-radius: 5px;
}
.titulo-seccion {
    color: #2B547E;
    margin-top: 1em;
    margin-bottom: 0.5em;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for tabs if not exists
if "last_report_id" not in st.session_state:
    st.session_state["last_report_id"] = ""
    
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["IoT Monitor", "Reportes PDF", "Preguntas de Ejemplo", "IoT ChatBot"])

# Tab 1: IoT Monitor Overview
with tab1:
    st.title("IoT Monitor: La Solución Integral para Dispositivos IoT")
    
    st.markdown("""
    <div class="highlight">
    <h2 class="titulo-seccion">¿Qué es IoT Monitor?</h2>
    <p>
        IoT Monitor es la plataforma definitiva para supervisar, gestionar y optimizar 
        tus dispositivos IoT, aprovechando <strong>Pinecone</strong> y <strong>OpenAI</strong>.
    </p>
    <h3 class="titulo-seccion">Beneficios Clave:</h3>
    <ul>
        <li><strong>Monitoreo en tiempo real</strong> (señal, batería, ubicación)</li>
        <li><strong>Búsqueda Inteligente</strong> con embeddings + re-rank</li>
        <li><strong>Reportes PDF</strong> para clientes</li>
        <li><strong>Alertas avanzadas</strong> (tamper detectado, restricción violada, etc.)</li>
    </ul>
    <p>
        Eleva la productividad de tus operaciones y ahorra costos con 
        una solución IoT verdaderamente completa. 
        ¡Bienvenido a <em>IoT Monitor</em>!
    </p>
    </div>
    """, unsafe_allow_html=True)

# Tab 2: PDF Reports
with tab2:
    logger.info("Entrando a la página Reportes")
    st.title("Generar Reportes PDF")
    
    tipo_consulta = st.radio("Tipo de consulta:", ("device_id", "user_id"))
    input_id = st.text_input(f"Ingrese {tipo_consulta}:", value=st.session_state["last_report_id"])
    
    if st.button("Generar reporte"):
        if not input_id.strip():
            st.warning("Por favor, ingresa un valor de ID válido.")
        else:
            st.session_state["last_report_id"] = input_id
    
            try:
                data = query_by_id(input_id, tipo_consulta)
                if data:
                    st.success(f"Se encontraron {len(data)} resultados para {tipo_consulta} = {input_id}")
                    for idx, item in enumerate(data, start=1):
                        st.subheader(f"Registro #{idx}")
                        st.json(item)
                        st.markdown("---")
    
                    pdf_bytes = generar_pdf(data)
                    if pdf_bytes:
                        st.download_button(
                            label="Descargar Reporte en PDF",
                            data=pdf_bytes,
                            file_name=f"results/Reporte_{tipo_consulta}_{input_id}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("No se generó el PDF.")
                else:
                    st.warning(f"No se encontraron registros para {tipo_consulta} = {input_id}")
            except Exception as e:
                st.error(f"Error consultando Pinecone: {e}")

# Tab 3: Example Questions
with tab3:
    logger.info("Entrando a la página Preguntas Ejemplo")
    st.title("Preguntas de Ejemplo - IoT Monitor")
    
    ejemplo_pregs = [
        "¿Cuál es la latitud del dispositivo 6cfc7a7a?",
        "¿Cuál es la longitud del dispositivo 6cfc7a7a?",
        "¿Cuál es el nivel de batería del dispositivo 58fc7458?",
        "¿Dónde se encuentra en este momento la persona con ID a4be2b7f en latitud?",
        "¿Dónde se encuentra en este momento la persona con ID a4be2b7f en longitud?",
        "¿Qué dispositivos tienen batería menor que 20%?",
        "¿Qué dispositivos tienen señal menor que -90 dBm?",
        "¿Qué usuarios tienen un dispositivo con tamper_detected=TRUE?",
        "¿Qué dispositivos han violado restricciones?",
        "¿Cuál es el estado del dispositivo 12ab34cd?",
        "¿Cuáles son las coordenadas de todos los dispositivos de la persona con ID ffff1234?",
        "¿Cuántos dispositivos están inactivos (status=0)?",
        "¿Hay dispositivos con batería muy baja?",
        "Muestra la última ubicación de user_id=abc123",
        "¿Quién tiene el nivel de batería más alto?",
        "¿Existen dispositivos con tamper_detected=TRUE y batería menor de 10%?",
        "¿Qué dispositivos se encuentran fuera de su zona permitida?",
        "¿Cuántos dispositivos han reportado restricción violada?",
        "¿Cuál es el promedio de señal en todos los dispositivos?",
        "¿Qué usuario tiene la señal más débil?",
        "¿Cuántos dispositivos en total están activos (status=1)?",
        "¿Cuáles son las coordenadas exactas del dispositivo ID 9999abcd?",
        "¿Qué nivel de batería tienen los dispositivos con tamper_detected=FALSE?",
        "¿Cuál es el usuario asociado al dispositivo con ID 5db3e12f?",
        "¿Se encuentra algún dispositivo con latitud mayor a 45?",
        "¿Qué dispositivo está más cerca del ecuador (latitud=0)?",
        "¿Cuáles dispositivos reportaron su última actualización el día 2025-01-25?",
        "¿Qué usuario sale con frecuencia de su zona restringida?",
        "¿Cómo saber la ubicación en tiempo real de un dispositivo?",
        "¿Cuál fue la última vez que se detectó tamper en el dispositivo 74fdc9b1?",
        "¿Cuántos dispositivos hay con señal por debajo de -100 dBm?",
        "¿Hay algún dispositivo con restricción_violation=TRUE en la ciudad X?",
        "Muestra todas las coordenadas de los dispositivos con batería < 15%.",
        "¿Qué dispositivos no han reportado manipulación (tamper_detected=FALSE)?",
        "¿Cuál es la persona con el mayor número de dispositivos activos?",
        "¿Podrías listar todos los dispositivos y sus estados?",
        "¿Algún dispositivo en la zona lat=30..31, lon=-95..-94?",
        "¿En qué fecha se actualizó por última vez el dispositivo 6cfc7a7a?",
        "¿Cuántos dispositivos tienen status=2?",
        "¿Hay algún dispositivo sin señal (signal_strength=Null o -999)?",
        "¿Existen usuarios con más de un dispositivo que reporta tamper_detected=TRUE?",
        "¿Cuántos dispositivos se encuentran con battery_level < 5% y status=1?",
        "¿Cuál es la persona con ID 056558c7 y dónde está su dispositivo?",
        "Listar todos los dispositivos que tengan restricción_violation=TRUE y lat>40",
        "¿Cómo saber si el dispositivo 6cfc7a7a cambió su estado recientemente?",
        "¿Cuáles son los últimos 5 registros del dispositivo 6cfc7a7a?",
        "Muestra la señal y la batería de cada dispositivo de user_id=f00dabcd",
        "¿Hay dispositivos con timestamp anterior a 2025-01-20?",
        "¿Qué dispositivos se han reactivado en la última hora?",
        "¿Cuáles son las coordenadas del usuario con ID abcd1234?",
        "¿Cuál es el device_id con menor nivel de batería?",
        "¿Cuántos dispositivos están operando normalmente?"
    ]

    # Group examples in columns for better visualization
    col1, col2 = st.columns(2)
    
    half_length = len(ejemplo_pregs) // 2
    with col1:
        for p in ejemplo_pregs[:half_length]:
            st.write(f"- {p}")
            
    with col2:
        for p in ejemplo_pregs[half_length:]:
            st.write(f"- {p}")

# Tab 4: IoT ChatBot
with tab4:
    logger.info("Entrando a la página ChatBot IoT con interfaz estilo ChatGPT")
    st.title("ChatBot IoT - Estilo ChatGPT")
    
    # Mostrar mensajes previos en burbujas de chat
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Crear una entrada de chat (input) en la parte inferior
    user_prompt = st.chat_input("Escribe tu pregunta o comando...")
    
    if user_prompt:  # cuando el usuario envía algo
        # Añadir el mensaje del usuario al historial
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        # Render burbuja
        with st.chat_message("user"):
            st.markdown(user_prompt)
    
        # 1) Interpretar la query => obtener contextos
        with st.spinner("Buscando información..."):
            contexts, used_filter, used_fallback = interpret_and_search(user_prompt)
    
        # 2) Generar respuesta
        with st.spinner("Generando respuesta..."):
            assistant_response = generate_chat_response(contexts, user_prompt)
    
        # Añadir respuesta al historial
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})
        # Render burbuja assistant
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
