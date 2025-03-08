"""
Caso 1: Aplicación de GenAI para la generación automática de SQL
Demuestra la generación de consultas SQL desde lenguaje natural para empresas de IoT.
"""
import streamlit as st
import os
import logging
import base64
from case_bigquery_sql.models.rag_bigquery_sql_system import RAGBigQuerySystem
from case_bigquery_sql.helpers.pinecone import initialize_pinecone_index
from case_bigquery_sql.helpers.report_generator import generate_pdf_from_df
from config import Config

logger = logging.getLogger(__name__)

def render_page():
    """Renderiza la página principal del Caso 1"""
    st.title("Caso 1 - Aplicación de GenAI para la generación de SQL")
    st.subheader("Caso de uso: Empresa de IoT")

    # Verificar configuraciones necesarias
    if not Config.validate_config():
        st.error("⚠️ Faltan configuraciones esenciales. Por favor configure las variables de entorno.")
        return

    # Crear tabs para organizar el contenido
    tab1, tab2, tab3 = st.tabs(["Información del Caso", "Generar Reporte", "Documentación"])

    # Tab 1: Información sobre el caso
    with tab1:
        render_info_tab()

    # Tab 2: Generación de reportes (funcionalidad principal)
    with tab2:
        render_report_tab()

    # Tab 3: Documentación
    with tab3:
        render_docs_tab()

def render_info_tab():
    """Renderiza la pestaña de información"""
    # Contenido descriptivo
    st.markdown("""
    ## Problema
    
    Los equipos de negocio y operaciones necesitan acceder a datos almacenados en bases de datos,
    pero no siempre tienen el conocimiento técnico para escribir consultas SQL complejas.
    
    ## Solución
    
    De acuerdo con las publicaciones de Gartner, la Inteligencia Artificial Generativa tiene un gran
    potencial para la automatización y personalización de servicios, permitiendo mejorar la eficiencia
    operativa del negocio y generar nuevas fuentes de ingreso.
    
    Este proyecto crea un "Banco de casos de usos de Inteligencia Artificial Generativa" enfocado
    en resolver problemas comunes en diferentes industrias que pueden ser automatizados.
    
    ### 2.1 Formulación del problema: ¿Qué se intenta resolver?
    
    Actualmente, la industria de tecnología enfrenta un proceso manual para generar
    reportes utilizando consultas directas a bases de datos y herramientas como Crystal
    Reports.
    
    ### 2.2 Contexto: ¿Por qué es importante resolver este problema?
    
    Resolver este problema es crucial porque:
    
    1. **Eficiencia operativa**: La generación manual de reportes consume tiempo que
       podría destinarse a actividades estratégicas.
    2. **Precisión**: El proceso manual tiene margen de error, afectando la toma de decisiones.
    3. **Escal
