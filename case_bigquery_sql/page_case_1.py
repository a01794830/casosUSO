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
    3. **Escalabilidad**: Con el crecimiento de la empresa, el volumen de datos y la
       demanda de reportes aumentarán, haciendo insostenible el enfoque actual.
    4. **Competitividad**: Automatizar este proceso con IA generativa permitirá a la
       empresa mantenerse a la vanguardia en el uso de tecnologías modernas,
       mejorando su capacidad de análisis y respuesta.
    """)

def render_report_tab():
    """Renderiza la pestaña de generación de reportes"""
    try:
        # Inicializar el sistema RAG conectando con Pinecone
        index_sql = initialize_pinecone_index("track-rag-sql")
        if not index_sql:
            st.error("⚠️ No se pudo conectar con Pinecone. Verifique su API key y configuración.")
            return
            
        rag = RAGBigQuerySystem(index_sql)
        st.header("Generar Reporte")
        
        # Selector de tipo de consulta
        query_type = st.radio(
            "Tipo de consulta:",
            ["Personalizada", "Dispositivos con batería baja", "Dispositivos con alertas", "Violaciones de restricciones"]
        )
        
        # Plantillas predefinidas
        templates = {
            "Dispositivos con batería baja": "Mostrar dispositivos con nivel de batería menor al 20%",
            "Dispositivos con alertas": "Listar dispositivos que tengan alertas activas",
            "Violaciones de restricciones": "Mostrar dispositivos que hayan violado restricciones geográficas"
        }
        
        # Campo de entrada de texto
        if query_type == "Personalizada":
            user_input = st.text_input(
                "Escribe el reporte que deseas generar:", 
                placeholder="e.g. Generar reporte de dispositivos en violación"
            )
        else:
            user_input = templates[query_type]
            st.info(f"Consulta predefinida: \"{user_input}\"")
            
        # Botones de acción
        col1, col2 = st.columns([1, 3])
        with col1:
            generate_button = st.button("Generar Reporte", use_container_width=True, type="primary")
        
        # Procesamiento al hacer clic en Generar
        if generate_button and user_input:
            with st.spinner("Generando SQL y preparando consulta..."):
                # Generar SQL
                sql_query = rag.generate_sql_query(user_input)
                
                # Mostrar SQL generado
                with st.expander("Ver consulta SQL generada", expanded=True):
                    st.code(sql_query, language="sql")
                
                # Ejecutar consulta
                try:
                    with st.spinner("Ejecutando consulta en BigQuery..."):
                        result_df = rag.query_bigquery(sql_query)
                        
                        if result_df is None or result_df.empty:
                            st.warning("La consulta no arrojó resultados.")
                            return
                    
                    # Mostrar resultados
                    st.subheader("Resultados")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Determinar si mostrar mapa
                    required_cols = ['Latitude', 'Longitude']
                    alt_required_cols = ['lat', 'lng']
                    template = "table_template.html"
                    has_map = False
                    
                    # Verificar columnas para mapa
                    if all(col in result_df.columns for col in required_cols):
                        st.subheader("Visualización en Mapa")
                        map_data = result_df[required_cols].rename(columns={'Latitude': 'lat', 'Longitude': 'lng'})
                        map_data_for_display = map_data.rename(columns={'lng': 'lon'})
                        st.map(map_data_for_display)
                        has_map = True
                        template = "map_template.html"
                    elif all(col in result_df.columns for col in alt_required_cols):
                        st.subheader("Visualización en Mapa")
                        map_data = result_df[alt_required_cols]
                        map_data_for_display = map_data.rename(columns={'lng': 'lon'})
                        st.map(map_data_for_display)
                        has_map = True
                        template = "map_template.html"
                    
                    # Generar PDF
                    with st.spinner("Generando reporte en PDF..."):
                        # Asegurar que exista el directorio
                        os.makedirs("results", exist_ok=True)
                        
                        # Ruta al template
                        template_path = os.path.join(os.getcwd(), "case_bigquery_sql", "templates", template)
                        
                        # Generar PDF
                        pdf_path = generate_pdf_from_df(
                            dataFrame=result_df,
                            template_path=template_path,
                            output_filename="results/report.pdf",
                            template_type="table" if not has_map else "map",
                            map_data={"locations": map_data.to_dict('records')} if has_map else None
                        )
                        
                        if pdf_path:
                            st.success("✅ Reporte generado exitosamente!")
                            
                            # Botón de descarga
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                with open(pdf_path, "rb") as file:
                                    pdfbytes = file.read()
                                    st.download_button(
                                        label="⬇️ Descargar Reporte PDF", 
                                        data=pdfbytes, 
                                        file_name=f"reporte_{query_type.replace(' ', '_')}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                            
                            # Vista previa
                            st.subheader("Vista previa del reporte:")
                            
                            # Embeber PDF para vista previa
                            with open(pdf_path, "rb") as file:
                                pdf_bytes = file.read()
                                st.markdown(f"""
                                    <embed src="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"
                                        width="100%" height="500" type="application/pdf">
                                    """, unsafe_allow_html=True)
                        else:
                            st.error("No se pudo generar el PDF.")
                
                except Exception as e:
                    st.error(f"Error al generar reporte: {str(e)}")
                    logger.error(f"Error en generación de reporte: {e}", exc_info=True)
    
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
        logger.error(f"Error inesperado en render_report_tab: {e}", exc_info=True)

def render_docs_tab():
    """Renderiza la pestaña de documentación"""
    st.header("Documentación")
    
    # Instrucciones de uso
    st.subheader("Instrucciones de uso")
    st.write("""
    1. Ve a la pestaña "Generar Reporte"
    2. Selecciona un tipo de consulta predefinida o elige "Personalizada"
    3. Si elegiste "Personalizada", escribe en lenguaje natural lo que deseas consultar
    4. Haz clic en "Generar Reporte" para ejecutar la consulta
    5. Visualiza los resultados, el mapa (si aplica) y descarga el PDF si lo necesitas
    """)
    
    # Ejemplos de consultas
    st.subheader("Ejemplos de consultas en lenguaje natural")
    
    example_queries = [
        "Generar reporte de dispositivos en violación de restricciones",
        "Mostrar dispositivos con batería menor al 15%",
        "Listar ubicaciones de dispositivos en Ciudad de México",
        "Mostrar dispositivos que no se han conectado en la última semana",
        "Encontrar dispositivos con señal débil y alertas de manipulación",
        "Listar usuarios cuyos dispositivos tienen nivel crítico de batería",
        "Mostrar dispositivos ordenados por nivel de batería de menor a mayor",
        "Encontrar dispositivos con coordenadas dentro del rango de latitud 19 a 20",
        "Mostrar los últimos 20 reportes de cada dispositivo",
        "Listar dispositivos con status inactivo desde hace más de 3 días"
    ]
    
    # Dos columnas para ejemplos
    col1, col2 = st.columns(2)
    
    # Distribuir ejemplos en columnas
    for i, example in enumerate(example_queries):
        if i < len(example_queries) // 2:
            with col1:
                if st.button(f"📋 {example}", key=f"example_{i}"):
                    # Guardar consulta en session state y redirigir
                    st.session_state['query_example'] = example
                    st.rerun()
        else:
            with col2:
                if st.button(f"📋 {example}", key=f"example_{i}"):
                    # Guardar consulta en session state y redirigir
                    st.session_state['query_example'] = example
                    st.rerun()
    
    # Documentación técnica
    with st.expander("Documentación técnica"):
        st.markdown("""
        ### Tecnologías utilizadas
        
        - **Streamlit**: Framework para la interfaz de usuario
        - **OpenAI**: Generación de embeddings y consultas SQL con IA
        - **Pinecone**: Almacenamiento vectorial para búsqueda semántica
        - **BigQuery**: Ejecución de consultas SQL y almacenamiento de datos
        - **PyMuPDF**: Generación de reportes PDF
        
        ### Detalles de implementación
        
        Este caso de uso implementa un sistema RAG (Retrieval Augmented Generation) que:
        
        1. Convierte la entrada en lenguaje natural a un embedding vectorial
        2. Busca ejemplos similares anteriores en Pinecone
        3. Usa estos ejemplos como contexto para que OpenAI genere SQL optimizado
        4. Ejecuta la consulta en BigQuery y visualiza los resultados
        5. Genera un reporte PDF con los hallazgos
        
        El sistema aprende con el tiempo ya que almacena cada consulta exitosa.
        """)

# Punto de entrada si se ejecuta directamente
if __name__ == "__main__":
    render_page()
