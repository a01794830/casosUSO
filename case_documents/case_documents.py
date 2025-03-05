"""
DocumentoRAG: Aplicación para análisis de documentos con IA generativa
"""
import streamlit as st
import logging
from case_documents.utils.doc_utils import parse_pdf, chunk_text
from case_documents.utils.pinecone_utils import upsert_docs, get_all_docs, search_docs
from case_documents.utils.summarizer import summarize_global_docs
from case_documents.utils.embedding_utils import generate_chat_response

from config import Config


st.title("Caso 4 - Inteligencia Artificial Generativa analysis de Documentos")
st.subheader("Caso de uso: DocumentoRAG")
multi = '''

'''
st.markdown(multi)

# Get project ID from environment variables
OPENAI_API_KEY = Config.OPENAI_API_KEY
PROJECT_ID = Config.BIGQUERY_PROJECT_ID

# Initialize session state variables
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
    
if "documents_indexed" not in st.session_state:
    st.session_state["documents_indexed"] = False
    
if "all_chunks" not in st.session_state:
    st.session_state["all_chunks"] = []

logger = logging.getLogger(__name__)

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Cargar Documentos", "Sumario Global", "Chat con Documentos"])

# Tab 1: Document Upload and Processing
with tab1:
    st.header("Cargar Documentos")
    st.write("Sube PDFs/TXT, se indexarán en Pinecone para preguntas/resúmenes.")

    uploaded_files = st.file_uploader("Selecciona archivos", 
                                      type=["pdf","txt"], 
                                      accept_multiple_files=True)
    
    if st.button("Procesar e Indexar"):
        if not uploaded_files:
            st.warning("No subiste archivos.")
            logger.warning("Ningún archivo subido.")
        else:
            with st.spinner("Procesando e indexando en Pinecone..."):
                all_chunks = []
                for f in uploaded_files:
                    if f.name.endswith(".pdf"):
                        text = parse_pdf(f)
                    else:
                        text = f.read().decode("utf-8", errors="ignore")

                    # Trocear
                    chunks = chunk_text(text, chunk_size=800)
                    all_chunks.extend(chunks)

                upsert_docs(all_chunks)
                st.session_state["documents_indexed"] = True
                st.session_state["all_chunks"] = all_chunks
                st.success(f"¡{len(all_chunks)} fragmentos indexados con éxito!")
                logger.info(f"Indexados {len(all_chunks)} trozos en Pinecone.")

# Tab 2: Global Summary
with tab2:
    st.header("Sumario Global")
    st.write("Crea un sumario de todos los documentos indexados en Pinecone.")
    
    if not st.session_state["documents_indexed"] and not st.button("Verificar documentos existentes"):
        st.info("Primero debes procesar e indexar documentos en la pestaña 'Cargar Documentos'.")
    else:
        if st.button("Generar Sumario"):
            with st.spinner("Obteniendo todos los trozos e intentando un sumario..."):
                docs = get_all_docs()
                if docs:
                    summary = summarize_global_docs(docs)
                    st.success("Sumario generado:")
                    st.write(summary)
                else:
                    st.warning("No se encontraron documentos indexados.")

# Tab 3: Chat with Documents
with tab3:
    st.header("Chat con Documentos")
    
    # Mostrar mensajes pasados
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Example questions in an expander
    with st.expander("Ejemplos de Preguntas"):
        example_questions = [
            "Resúmeme el capítulo 1",
            "Dame 3 conclusiones principales del PDF subido",
            "Explica la diferencia entre X y Y en el texto",
            "¿Qué metodología se describe en la sección 2?",
            "¿Cuáles son las conclusiones finales del autor?",
            "¿Qué ejemplos brinda para sustentar su argumento?",
            "Haz un resumen corto del documento (máx. 50 palabras)",
            "¿Qué limitaciones menciona el artículo?",
            "Compara la introducción y la discusión",
            "Dame 5 puntos clave del texto",
            "Explica brevemente la sección 'Resultados'",
            "¿Cómo define el autor el concepto de 'innovación'?",
            "¿Qué referencias son más relevantes?",
            "Resume el segundo capítulo en 3 oraciones",
            "¿Cuál es la postura principal del autor frente a X?",
            "¿Qué recomendaciones propone el texto al final?",
            "Dime una cita textual que respalde la hipótesis",
            "¿Cómo se relacionan los hallazgos con la teoría inicial?",
            "Explica la diferencia metodológica entre 'cuantitativo' y 'cualitativo' en el texto",
            "¿Qué implicaciones prácticas menciona para futuros estudios?"
        ]
        for q in example_questions:
            st.write(f"- {q}")
    
    if not st.session_state["documents_indexed"]:
        st.info("Primero debes procesar e indexar documentos en la pestaña 'Cargar Documentos'.")
    
    user_input = st.chat_input("Pregunta o pide un resumen...")

    if user_input:
        # user
        st.session_state["chat_messages"].append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Buscar en Pinecone
        docs = search_docs(user_input)
        logger.info(f"Encontrados {len(docs)} docs para la query.")

        # Generar respuesta
        assistant_resp = generate_chat_response(docs, user_input)
        st.session_state["chat_messages"].append({"role":"assistant","content":assistant_resp})

        with st.chat_message("assistant"):
            st.markdown(assistant_resp)
