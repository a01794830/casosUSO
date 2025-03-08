"""
Caso 2: Análisis de Datos IoT con IA Generativa
Demuestra la aplicación de IA generativa para monitoreo y análisis de dispositivos IoT.
"""
import streamlit as st
import logging
import os
import base64
from datetime import datetime
from typing import Dict, List, Any
from case_iot.utils.pinecone_utils import query_by_id, interpret_and_search
from case_iot.utils.embedding_utils import generate_chat_response
from case_iot.utils.pdf_utils import generar_pdf
from config import Config

# Configurar logging
logger = logging.getLogger(__name__)

def render_iot_monitor_app():
    """
    Función principal que renderiza la aplicación IoT Monitor.
    """
    # Título principal
    st.title("Caso 2 - Inteligencia Artificial Generativa para IoT")
    
    # Verificar configuraciones API
    if not Config.OPENAI_API_KEY or not Config.PINECONE_API_KEY:
        st.error("⚠️ Faltan configuraciones de API. Por favor configure las variables de entorno.")
        return
    
    # CSS personalizado para mejorar la interfaz
    st.markdown("""
    <style>
    .highlight {
        background-color: #f0f7fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2C5BA9;
        margin-bottom: 1rem;
    }
    .titulo-seccion {
        color: #2B547E;
        margin-top: 1em;
        margin-bottom: 0.5em;
        font-weight: 600;
    }
    .alert-success {
        background-color: #e6f3e6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
    }
    .alert-warning {
        background-color: #fff8e6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff9800;
    }
    .alert-danger {
        background-color: #fde8e8;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #f44336;
    }
    .device-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Inicializar variables de estado de sesión
    if "last_report_id" not in st.session_state:
        st.session_state["last_report_id"] = ""
        
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    if "settings" not in st.session_state:
        st.session_state["settings"] = {
            "model": "gpt-3.5-turbo",
            "map_type": "street",
            "language": "es"
        }
    
    # Crear pestañas para diferentes funcionalidades
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 IoT Monitor", 
        "📊 Reportes PDF", 
        "📝 Ejemplos", 
        "💬 IoT ChatBot"
    ])
    
    # Pestaña 1: Visión general del IoT Monitor
    with tab1:
        render_overview_tab()
    
    # Pestaña 2: Generación de reportes PDF
    with tab2:
        render_reports_tab()
    
    # Pestaña 3: Ejemplos de consultas
    with tab3:
        render_examples_tab()
    
    # Pestaña 4: ChatBot IoT
    with tab4:
        render_chatbot_tab()

def render_overview_tab():
    """
    Renderiza la pestaña de visión general del IoT Monitor.
    """
    st.title("IoT Monitor: La Solución Integral para Dispositivos IoT")
    
    # Descripción principal con formato mejorado
    st.markdown("""
    <div class="highlight">
    <h2 class="titulo-seccion">¿Qué es IoT Monitor?</h2>
    <p>
        IoT Monitor es la plataforma definitiva para supervisar, gestionar y optimizar 
        tus dispositivos IoT, aprovechando tecnologías avanzadas como <strong>Pinecone</strong> para
        almacenamiento vectorial y <strong>OpenAI</strong> para procesamiento de lenguaje natural.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Características y beneficios
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <h3 class="titulo-seccion">Monitoreo en Tiempo Real</h3>
        <ul>
            <li>Seguimiento continuo de señal y batería</li>
            <li>Geolocalización precisa de dispositivos</li>
            <li>Detección de manipulaciones y violaciones</li>
            <li>Alertas automatizadas y personalizables</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <h3 class="titulo-seccion">Gestión Inteligente de Datos</h3>
        <ul>
            <li>Búsquedas semánticas con embeddings vectoriales</li>
            <li>Filtrado avanzado mediante IA</li>
            <li>Procesamiento de consultas en lenguaje natural</li>
            <li>Re-ranking para resultados más precisos</li>
        </ul>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <h3 class="titulo-seccion">Informes y Análisis</h3>
        <ul>
            <li>Generación automatizada de reportes PDF</li>
            <li>Visualizaciones geoespaciales</li>
            <li>Estadísticas de rendimiento</li>
            <li>Identificación proactiva de problemas</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <h3 class="titulo-seccion">Experiencia de Usuario Optimizada</h3>
        <ul>
            <li>Interfaz conversacional con ChatBot especializado</li>
            <li>Consultas en lenguaje natural</li>
            <li>Respuestas contextualizadas y precisas</li>
            <li>Interfaz intuitiva y accesible</li>
        </ul>
        """, unsafe_allow_html=True)
    
    # Diagrama de arquitectura o imagen representativa
    st.subheader("Arquitectura del Sistema")
    
    st.markdown("""
    ```
    +-------------------+     +-------------------+     +-------------------+
    |                   |     |                   |     |                   |
    |  Dispositivos IoT | --> | API/Gateway       | --> |  Procesamiento   |
    |  (Sensores)       |     | (Recepción datos) |     |  (Pinecone/OpenAI)|
    |                   |     |                   |     |                   |
    +-------------------+     +-------------------+     +-------------------+
                                                                |
                                                                v
    +-------------------+     +-------------------+     +-------------------+
    |                   |     |                   |     |                   |
    |  Interfaz Usuario | <-- | Generación        | <-- |  Análisis         |
    |  (Streamlit)      |     | (Reportes/Mapas)  |     |  (Embeddings/RAG) |
    |                   |     |                   |     |                   |
    +-------------------+     +-------------------+     +-------------------+
    ```
    """)
    
    # Casos de uso
    st.subheader("Principales Casos de Uso")
    
    cases = [
        {
            "title": "Monitoreo de Dispositivos de Rastreo",
            "description": "Seguimiento de dispositivos GPS para logística, flotas o seguridad, permitiendo localizar activos y detectar violaciones de perímetro."
        },
        {
            "title": "Gestión de Alertas",
            "description": "Identificación temprana de dispositivos con batería baja, señal débil o posibles manipulaciones para intervención preventiva."
        },
        {
            "title": "Análisis de Patrones",
            "description": "Detección de comportamientos inusuales y patrones que pueden indicar problemas o áreas de mejora."
        },
        {
            "title": "Reportes Automatizados",
            "description": "Generación de informes detallados sobre estado y rendimiento de dispositivos para equipos internos o clientes."
        }
    ]
    
    # Mostrar casos de uso en formato de tarjetas
    cols = st.columns(2)
    for i, case in enumerate(cases):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="device-card">
                <h4>{case['title']}</h4>
                <p>{case['description']}</p>
            </div>
            """, unsafe_allow_html=True)

def render_reports_tab():
    """
    Renderiza la pestaña de generación de reportes PDF.
    """
    logger.info("Entrando a la página Reportes")
    st.title("Generar Reportes PDF")
    
    # Opciones de consulta
    col1, col2 = st.columns(2)
    
    with col1:
        tipo_consulta = st.radio("Tipo de consulta:", ("device_id", "user_id"))
    
    with col2:
        input_id = st.text_input(
            f"Ingrese {tipo_consulta}:", 
            value=st.session_state["last_report_id"],
            placeholder=f"Ej: 6cfc7a7a para device_id"
        )
    
    # Botón de generación
    generate_col, status_col = st.columns([1, 2])
    with generate_col:
        generar_button = st.button(
            "📊 Generar Reporte", 
            type="primary", 
            use_container_width=True,
            disabled=not input_id.strip()
        )
    
    # Procesar generación de reporte
    if generar_button:
        if not input_id.strip():
            st.warning("Por favor, ingresa un valor de ID válido.")
        else:
            # Guardar ID para futuros usos
            st.session_state["last_report_id"] = input_id
            
            try:
                # Consultar datos
                with st.spinner(f"Consultando datos para {tipo_consulta} = {input_id}..."):
                    data = query_by_id(input_id, tipo_consulta)
                
                if data:
                    st.success(f"✅ Se encontraron {len(data)} registros para {tipo_consulta} = {input_id}")
                    
                    # Mostrar datos en formato de tarjetas
                    with st.expander("Ver detalles de registros", expanded=True):
                        for idx, item in enumerate(data, start=1):
                            st.markdown(f"""
                            <div class="device-card">
                                <h4>Registro #{idx}</h4>
                                <pre>{format_device_data(item)}</pre>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Generar PDF
                    with st.spinner("Generando reporte PDF..."):
                        pdf_bytes = generar_pdf(
                            data,
                            titulo=f"Reporte IoT: {tipo_consulta}={input_id}"
                        )
                        
                        if pdf_bytes:
                            # Asegurar que exista el directorio
                            os.makedirs("results", exist_ok=True)
                            
                            # Guardar PDF localmente
                            pdf_filename = f"Reporte_{tipo_consulta}_{input_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            pdf_path = os.path.join("results", pdf_filename)
                            
                            with open(pdf_path, "wb") as f:
                                f.write(pdf_bytes)
                            
                            # Botón de descarga
                            st.download_button(
                                label="⬇️ Descargar Reporte en PDF",
                                data=pdf_bytes,
                                file_name=pdf_filename,
                                mime="application/pdf"
                            )
                            
                            # Vista previa
                            st.subheader("Vista previa del PDF")
                            st.markdown(f"""
                                <embed src="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"
                                    width="100%" height="500" type="application/pdf">
                                """, unsafe_allow_html=True)
                            
                        else:
                            st.error("❌ No se pudo generar el PDF.")
                else:
                    st.warning(f"⚠️ No se encontraron registros para {tipo_consulta} = {input_id}")
                    
            except Exception as e:
                logger.error(f"Error consultando Pinecone: {e}", exc_info=True)
                st.error(f"❌ Error consultando datos: {str(e)}")

def format_device_data(item: Dict[str, Any]) -> str:
    """
    Formatea datos de dispositivo para visualización.
    
    Args:
        item (Dict[str, Any]): Datos del dispositivo
        
    Returns:
        str: Texto formateado
    """
    # Agrupar campos por categoría
    id_fields = ["device_id", "user_id"]
    location_fields = ["latitude", "longitude"]
    status_fields = ["battery_level", "signal_strength", "tamper_detected", "status", "restriction_violation"]
    time_fields = ["timestamp"]
    
    # Construir texto formateado
    lines = []
    
    # Sección de identificación
    id_section = []
    for field in id_fields:
        if field in item:
            id_section.append(f"{field}: {item[field]}")
    if id_section:
        lines.append("📱 Identificación:")
        lines.extend([f"  {line}" for line in id_section])
    
    # Sección de ubicación
    location_section = []
    for field in location_fields:
        if field in item:
            location_section.append(f"{field}: {item[field]}")
    if location_section:
        lines.append("\n📍 Ubicación:")
        lines.extend([f"  {line}" for line in location_section])
    
    # Sección de estado
    status_section = []
    for field in status_fields:
        if field in item:
            # Formateo especial según campo
            if field == "battery_level":
                value = item[field]
                status_section.append(f"{field}: {value}% {'🔋' if float(value) > 50 else '🪫'}")
            elif field == "tamper_detected" and str(item[field]).lower() in ["true", "1"]:
                status_section.append(f"{field}: {item[field]} ⚠️")
            elif field == "restriction_violation" and str(item[field]).lower() in ["true", "1"]:
                status_section.append(f"{field}: {item[field]} 🚨")
            else:
                status_section.append(f"{field}: {item[field]}")
    if status_section:
        lines.append("\n🔄 Estado:")
        lines.extend([f"  {line}" for line in status_section])
    
    # Sección de tiempo
    time_section = []
    for field in time_fields:
        if field in item:
            time_section.append(f"{field}: {item[field]}")
    if time_section:
        lines.append("\n🕒 Tiempo:")
        lines.extend([f"  {line}" for line in time_section])
    
    # Otros campos
    other_fields = [f for f in item.keys() if f not in id_fields + location_fields + status_fields + time_fields]
    if other_fields:
        lines.append("\n📄 Otros datos:")
        for field in other_fields:
            lines.append(f"  {field}: {item[field]}")
    
    return "\n".join(lines)

def render_examples_tab():
    """
    Renderiza la pestaña de ejemplos de consultas.
    """
    logger.info("Entrando a la página de Ejemplos")
    st.title("Ejemplos de Consultas para IoT Monitor")
    
    # Agrupar ejemplos por categorías
    categories = {
        "Consultas de dispositivos": [
            "¿Cuál es la latitud del dispositivo 6cfc7a7a?",
            "¿Cuál es la longitud del dispositivo 6cfc7a7a?",
            "¿Cuál es el nivel de batería del dispositivo 58fc7458?",
            "¿Cuál es el estado del dispositivo 12ab34cd?",
            "¿Cuáles son las coordenadas exactas del dispositivo ID 9999abcd?",
            "¿Cómo saber la ubicación en tiempo real de un dispositivo?",
            "¿Cuál fue la última vez que se detectó tamper en el dispositivo 74fdc9b1?",
            "¿Cómo saber si el dispositivo 6cfc7a7a cambió su estado recientemente?",
            "¿Cuáles son los últimos 5 registros del dispositivo 6cfc7a7a?"
        ],
        "Consultas de usuarios": [
            "¿Dónde se encuentra en este momento la persona con ID a4be2b7f en latitud?",
            "¿Dónde se encuentra en este momento la persona con ID a4be2b7f en longitud?",
            "¿Cuáles son las coordenadas de todos los dispositivos de la persona con ID ffff1234?",
            "Muestra la última ubicación de user_id=abc123",
            "¿Quién tiene el nivel de batería más alto?",
            "¿Qué usuario tiene la señal más débil?",
            "¿Cuál es el usuario asociado al dispositivo con ID 5db3e12f?",
            "¿Cuál es la persona con ID 056558c7 y dónde está su dispositivo?",
            "Muestra la señal y la batería de cada dispositivo de user_id=f00dabcd"
        ],
        "Consultas de alertas y problemas": [
            "¿Qué dispositivos tienen batería menor que 20%?",
            "¿Qué dispositivos tienen señal menor que -90 dBm?",
            "¿Qué usuarios tienen un dispositivo con tamper_detected=TRUE?",
            "¿Qué dispositivos han violado restricciones?",
            "¿Hay dispositivos con batería muy baja?",
            "¿Existen dispositivos con tamper_detected=TRUE y batería menor de 10%?",
            "¿Qué dispositivos se encuentran fuera de su zona permitida?",
            "¿Cuántos dispositivos han reportado restricción violada?",
            "¿Hay algún dispositivo con restricción_violation=TRUE en la ciudad X?",
            "¿Existen usuarios con más de un dispositivo que reporta tamper_detected=TRUE?",
            "¿Cuántos dispositivos se encuentran con battery_level < 5% y status=1?"
        ],
        "Consultas estadísticas": [
            "¿Hay dispositivos con batería muy baja?",
            "¿Cuántos dispositivos están inactivos (status=0)?",
            "¿Cuál es el promedio de señal en todos los dispositivos?",
            "¿Cuántos dispositivos en total están activos (status=1)?",
            "¿Se encuentra algún dispositivo con latitud mayor a 45?",
            "¿Qué dispositivo está más cerca del ecuador (latitud=0)?",
            "¿Hay algún dispositivo sin señal (signal_strength=Null o -999)?",
            "¿Cuántos dispositivos tienen status=2?",
            "¿Qué dispositivos se han reactivado en la última hora?",
            "¿Cuál es el device_id con menor nivel de batería?",
            "¿Cuántos dispositivos están operando normalmente?"
        ]
    }
    
    # Mostrar ejemplos por categoría
    for category, examples in categories.items():
        st.subheader(category)
        
        # Crear columnas para ejemplos
        cols = st.columns(2)
        half_length = len(examples) // 2 + (len(examples) % 2)
        
        # Distribuir ejemplos en columnas
        for i, example in enumerate(examples):
            col_idx = 0 if i < half_length else 1
            with cols[col_idx]:
                # Botón que se puede usar para probar el ejemplo
                if st.button(f"🔍 {example}", key=f"example_{category}_{i}"):
                    # Guardar consulta en session state y redirigir a pestaña de chatbot
                    st.session_state["chat_query"] = example
                    st.session_state["active_tab"] = "ChatBot"
                    st.rerun()

def render_chatbot_tab():
    """
    Renderiza la pestaña del ChatBot IoT.
    """
    logger.info("Entrando a la página ChatBot IoT")
    st.title("ChatBot IoT - Consulta Inteligente")
    
    # Opciones de configuración en la barra lateral
    with st.sidebar:
        st.subheader("Configuración del ChatBot")
        
        # Modelo de IA
        model = st.selectbox(
            "Modelo:",
            ["gpt-3.5-turbo", "gpt-4o"],
            index=0
        )
        
        # Guardar configuración
        if model != st.session_state["settings"]["model"]:
            st.session_state["settings"]["model"] = model
        
        # Botón para limpiar historial
        if st.button("🗑️ Limpiar historial", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    
    # Contenedor para mensajes de chat
    chat_container = st.container()
    
    # Mostrar mensajes previos
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Procesar consulta guardada desde otra pestaña
    if "chat_query" in st.session_state and st.session_state["chat_query"]:
        user_prompt = st.session_state["chat_query"]
        st.session_state["chat_query"] = ""  # Limpiar para evitar repetición
    else:
        # Entrada de chat normal
        user_prompt = st.chat_input("Escribe tu pregunta sobre dispositivos IoT...")
    
    # Procesar la entrada del usuario
    if user_prompt:
        # Añadir mensaje del usuario al historial
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        
        # Mostrar mensaje del usuario
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_prompt)
        
        # 1) Interpretar la consulta y obtener contextos relevantes
        with st.spinner("Buscando información relevante..."):
            contexts, used_filter, used_fallback = interpret_and_search(user_prompt)
        
        # 2) Generar respuesta
        with st.spinner("Generando respuesta..."):
            # Usar el modelo configurado
            model = st.session_state["settings"]["model"]
            assistant_response = generate_chat_response(contexts, user_prompt, model=model)
        
        # Añadir respuesta al historial
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})
        
        # Mostrar respuesta del asistente
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            
            # Mostrar información sobre el proceso (opcional)
            with st.expander("Detalles de la consulta", expanded=False):
                st.markdown(f"""
                - **Modelo utilizado**: {model}
                - **Consulta interpretada automáticamente**: {"✅ Sí" if used_filter else "❌ No"}
                - **Método de búsqueda**: {"🔍 Filtrado directo" if used_filter else "🔍 Búsqueda por similitud"} 
                - **Fallback activado**: {"✅ Sí" if used_fallback else "❌ No"}
                - **Contextos relevantes encontrados**: {len(contexts)}
                """)
                
                if contexts:
                    st.markdown("**Fragmentos de contexto utilizados:**")
                    for i, ctx in enumerate(contexts[:3]):  # Mostrar solo los 3 primeros
                        st.markdown(f"**Fragmento {i+1}:**")
                        st.code(ctx[:300] + "..." if len(ctx) > 300 else ctx)

# Función auxiliar para formatar datos
def format_json_data(data):
    """
    Formatea datos JSON para mostrarlos de forma más amigable.
    
    Args:
        data: Datos JSON a formatear
        
    Returns:
        str: Representación formateada
    """
    import json
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except:
        return str(data)

# Punto de entrada si se ejecuta directamente
if __name__ == "__main__":
    render_iot_monitor_app()
