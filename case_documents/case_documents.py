"""
Aplicación DocumentoRAG: Análisis de documentos con IA generativa.
Permite cargar, indexar, resumir y consultar documentos usando RAG.
"""
import streamlit as st
import logging
import os
import time
from typing import List, Dict, Any, Tuple
from case_documents.utils.doc_utils import parse_pdf, chunk_text
from case_documents.utils.pinecone_utils import upsert_docs, get_all_docs, search_docs
from case_documents.utils.summarizer import summarize_global_docs
from case_documents.utils.embedding_utils import generate_chat_response
from config import Config

# Configurar logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def render_documento_rag_app():
    """
    Función principal que renderiza la aplicación DocumentoRAG.
    """
    # Título y descripción de la aplicación
    st.title("Caso 4 - Análisis de Documentos con IA Generativa")
    st.subheader("Caso de uso: DocumentoRAG")
    
    # Verificar configuraciones API
    if not Config.OPENAI_API_KEY or not Config.PINECONE_API_KEY:
        st.error("⚠️ Faltan configuraciones de API. Por favor configure las variables de entorno OPENAI_API_KEY y PINECONE_API_KEY.")
        return
    
    # Inicializar variables de estado de sesión
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
        
    if "documents_indexed" not in st.session_state:
        st.session_state["documents_indexed"] = False
        
    if "all_chunks" not in st.session_state:
        st.session_state["all_chunks"] = []
        
    if "index_stats" not in st.session_state:
        st.session_state["index_stats"] = {"num_docs": 0, "last_update": None}

    # Crear pestañas para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["Cargar Documentos", "Resumen Global", "Chat con Documentos"])

    # Pestaña 1: Carga y procesamiento de documentos
    with tab1:
        render_document_upload_tab()

    # Pestaña 2: Resumen global
    with tab2:
        render_summary_tab()

    # Pestaña 3: Chat con documentos
    with tab3:
        render_chat_tab()

def render_document_upload_tab():
    """
    Renderiza la pestaña de carga de documentos.
    """
    st.header("Cargar Documentos")
    st.write("Sube archivos PDF o TXT para indexarlos en la base de conocimiento.")

    # Opciones de configuración
    with st.expander("Opciones de procesamiento", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Tamaño de fragmento (caracteres)", 
                                         min_value=200, max_value=2000, value=800, step=100,
                                         help="Tamaño aproximado de cada fragmento en caracteres")
        with col2:
            overlap = st.number_input("Superposición (caracteres)", 
                                     min_value=0, max_value=500, value=100, step=50,
                                     help="Número de caracteres que se superponen entre fragmentos")

    # Selector de archivos
    uploaded_files = st.file_uploader("Selecciona archivos", 
                                     type=["pdf", "txt"], 
                                     accept_multiple_files=True,
                                     help="Formatos soportados: PDF, TXT")
    
    # Botón de procesamiento
    process_col, status_col = st.columns([1, 2])
    with process_col:
        process_button = st.button("📋 Procesar e Indexar", 
                                  type="primary", 
                                  use_container_width=True,
                                  disabled=not uploaded_files)
    
    # Procesar documentos al hacer clic
    if process_button:
        if not uploaded_files:
            st.warning("⚠️ No has subido ningún archivo.")
            logger.warning("Intento de procesamiento sin archivos subidos")
        else:
            # Mostrar progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Procesar cada archivo
                all_chunks = []
                for i, file in enumerate(uploaded_files):
                    # Actualizar estado
                    status_text.text(f"Procesando archivo {i+1}/{len(uploaded_files)}: {file.name}")
                    progress_bar.progress((i * 2) / (len(uploaded_files) * 2 + 1))
                    
                    # Extraer texto según formato
                    if file.name.lower().endswith(".pdf"):
                        text = parse_pdf(file)
                    else:  # Asumimos TXT
                        text = file.read().decode("utf-8", errors="ignore")
                        
                    # Mostrar tamaño del texto extraído
                    logger.info(f"Extraídos {len(text)} caracteres de {file.name}")
                    
                    # Actualizar progreso
                    status_text.text(f"Fragmentando {file.name}...")
                    progress_bar.progress((i * 2 + 1) / (len(uploaded_files) * 2 + 1))
                    
                    # Fragmentar texto
                    file_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                    all_chunks.extend(file_chunks)
                    
                # Indexar todos los fragmentos
                status_text.text(f"Indexando {len(all_chunks)} fragmentos en Pinecone...")
                progress_bar.progress(len(uploaded_files) * 2 / (len(uploaded_files) * 2 + 1))
                
                # Insertar en Pinecone
                success = upsert_docs(all_chunks)
                
                # Actualizar estado
                progress_bar.progress(1.0)
                if success:
                    # Actualizar estado de sesión
                    st.session_state["documents_indexed"] = True
                    st.session_state["all_chunks"] = all_chunks
                    st.session_state["index_stats"] = {
                        "num_docs": len(all_chunks),
                        "last_update": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Mostrar éxito
                    status_text.text("✅ Procesamiento completado")
                    st.success(f"¡{len(all_chunks)} fragmentos indexados con éxito!")
                    
                    # Mostrar estadísticas
                    with st.expander("Detalles de procesamiento", expanded=True):
                        st.markdown(f"""
                        - **Documentos procesados:** {len(uploaded_files)}
                        - **Fragmentos generados:** {len(all_chunks)}
                        - **Tamaño promedio:** {sum(len(c) for c in all_chunks) // len(all_chunks) if all_chunks else 0} caracteres/fragmento
                        - **Fecha de indexación:** {st.session_state["index_stats"]["last_update"]}
                        """)
                else:
                    status_text.text("❌ Error en la indexación")
                    st.error("No se pudieron indexar los documentos. Revisa los logs para más detalles.")
                
            except Exception as e:
                logger.error(f"Error en procesamiento de documentos: {str(e)}", exc_info=True)
                st.error(f"Error: {str(e)}")
                status_text.text("❌ Error en el procesamiento")

def render_summary_tab():
    """
    Renderiza la pestaña de resumen global.
    """
    st.header("Resumen Global")
    st.write("Genera un resumen de todos los documentos indexados en la base de conocimiento.")
    
    # Verificar si hay documentos indexados
    if not st.session_state["documents_indexed"]:
        # Botón para verificar documentos existentes
        check_docs = st.button("🔍 Verificar documentos existentes", use_container_width=True)
        
        if check_docs:
            with st.spinner("Buscando documentos indexados..."):
                docs = get_all_docs()
                if docs:
                    st.session_state["documents_indexed"] = True
                    st.session_state["all_chunks"] = docs
                    st.session_state["index_stats"] = {
                        "num_docs": len(docs),
                        "last_update": "Desconocido (documentos pre-existentes)"
                    }
                    st.success(f"Se encontraron {len(docs)} fragmentos indexados previamente.")
                    st.rerun()
                else:
                    st.warning("No se encontraron documentos indexados.")
        else:
            st.info("Primero debes procesar e indexar documentos en la pestaña 'Cargar Documentos'.")
    else:
        # Mostrar estadísticas de documentos
        st.caption(f"Hay {st.session_state['index_stats']['num_docs']} fragmentos indexados (última actualización: {st.session_state['index_stats']['last_update']})")
        
        # Opciones de resumen
        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox("Modelo:", ["gpt-3.5-turbo", "gpt-4o"], index=0)
        with col2:
            max_length = st.slider("Longitud máxima:", min_value=100, max_value=1000, value=500, step=50)
        
        # Botón de generación
        if st.button("📝 Generar Resumen", type="primary", use_container_width=True):
            with st.spinner("Obteniendo fragmentos y generando resumen..."):
                # Obtener todos los fragmentos (o usar los ya cargados)
                if st.session_state["all_chunks"]:
                    docs = st.session_state["all_chunks"]
                else:
                    docs = get_all_docs()
                
                if not docs:
                    st.warning("No se encontraron documentos indexados.")
                    return
                
                # Generar resumen
                summary = summarize_global_docs(docs, model=model, max_tokens=max_length)
                
                # Mostrar resumen
                st.success("✅ Resumen generado correctamente")
                
                st.subheader("Resumen Global")
                st.markdown(summary)
                
                # Opción de descarga
                summary_download = f"""# Resumen Global de Documentos
Generado el {time.strftime("%Y-%m-%d %H:%M:%S")}

{summary}
"""
                st.download_button(
                    label="⬇️ Descargar Resumen", 
                    data=summary_download, 
                    file_name="resumen_global.md",
                    mime="text/markdown"
                )

def render_chat_tab():
    """
    Renderiza la pestaña de chat con documentos.
    """
    st.header("Chat con Documentos")
    st.write("Haz preguntas sobre el contenido de tus documentos indexados.")
    
    # Verificar si hay documentos indexados
    if not st.session_state["documents_indexed"]:
        check_docs = st.button("🔍 Verificar documentos existentes", key="check_docs_chat", use_container_width=True)
        
        if check_docs:
            with st.spinner("Buscando documentos indexados..."):
                docs = get_all_docs()
                if docs:
                    st.session_state["documents_indexed"] = True
                    st.session_state["all_chunks"] = docs
                    st.session_state["index_stats"] = {
                        "num_docs": len(docs),
                        "last_update": "Desconocido (documentos pre-existentes)"
                    }
                    st.success(f"Se encontraron {len(docs)} fragmentos indexados previamente.")
                    st.rerun()
                else:
                    st.warning("No se encontraron documentos indexados.")
            
        st.info("Primero debes procesar e indexar documentos en la pestaña 'Cargar Documentos'.")
        return
    
    # Interfaz de chat
    st.caption(f"Hay {st.session_state['index_stats']['num_docs']} fragmentos indexados disponibles para consulta")
    
    # Opciones de modelo
    with st.sidebar:
        st.subheader("Configuración del Chat")
        model = st.selectbox("Modelo:", ["gpt-3.5-turbo", "gpt-4o"], index=0)
        
        # Ejemplos de preguntas
        st.subheader("Ejemplos de Preguntas")
        example_questions = [
            "Resume el contenido principal.",
            "¿Cuáles son las conclusiones principales?",
            "Explica el concepto de X mencionado en el documento.",
            "¿Qué metodología se describe?",
            "¿Cuál es la postura del autor sobre Y?",
            "Resume la sección sobre Z.",
            "¿Cómo se comparan A y B en el texto?",
            "Enumera los 3 puntos clave del documento."
        ]
        
        for q in example_questions:
            if st.button(f"📝 {q}", key=f"example_{q}", use_container_width=True):
                # Establecer como mensaje de usuario
                st.session_state["chat_input"] = q
                # Forzar recargar para procesar
                st.rerun()
    
    # Mostrar mensajes previos
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Campo de entrada de chat
    chat_input_key = "chat_input"
    user_input = st.chat_input("Pregunta o pide un resumen sobre tus documentos...", key=chat_input_key)
    
    # Procesar la entrada del usuario (ya sea de input directo o de botón de ejemplo)
    if user_input or (chat_input_key in st.session_state and st.session_state[chat_input_key]):
        # Obtener mensaje (de input directo o de state)
        if user_input:
            query = user_input
        else:
            query = st.session_state[chat_input_key]
            # Limpiar después de usar
            st.session_state[chat_input_key] = ""
        
        # Añadir mensaje del usuario
        st.session_state["chat_messages"].append({"role": "user", "content": query})
        
        # Mostrar mensaje del usuario
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)
        
        # Buscar documentos relevantes
        with st.spinner("Buscando información relevante..."):
            docs = search_docs(query)
            
            if not docs:
                st.warning("No se encontraron documentos relevantes para tu consulta.")
                assistant_resp = "No encontré información relevante en los documentos para responder a tu consulta. Por favor, intenta reformular tu pregunta o verifica que los documentos contengan la información que buscas."
            else:
                # Generar respuesta
                assistant_resp = generate_chat_response(docs, query, model=model)
        
        # Añadir respuesta del asistente
        st.session_state["chat_messages"].append({"role": "assistant", "content": assistant_resp})
        
        # Mostrar respuesta
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(assistant_resp)
        
        # Mostrar fuentes (opcional)
        with st.expander("Ver fuentes consultadas", expanded=False):
            st.caption("Fragmentos relevantes utilizados para generar la respuesta:")
            for i, doc in enumerate(docs[:3]):  # Mostrar solo los 3 primeros
                st.markdown(f"**Fragmento {i+1}:**")
                st.markdown(f"```\n{doc[:300]}...\n```")

# Punto de entrada principal
if __name__ == "__main__":
    render_documento_rag_app()
