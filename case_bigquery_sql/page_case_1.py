"""
    Use Case 3 
"""
import streamlit as st
import os
from case_bigquery_sql.models.rag_bigquery_sql_system import RAGBigQuerySystem
from case_bigquery_sql.helpers.pinecone import initialize_pinecone_index
from case_bigquery_sql.helpers.report_generator import generate_pdf_from_df
import base64
import json

st.title("Caso 1 - Aplicacion de GenAI para la generacion de SQL")
st.subheader("Caso de uso: Empresa de IoT")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Información del Caso", "Generar Reporte", "Documentación"])

# Tab 1: Information about the case
with tab1:
    multi = '''
    Problema:
    - Equipos de desarollo pueden simplificar y proveer pormedio de 


    De acuerdo con las publicaciones de Gartner [1], el potencial que la Inteligencia Artificial
    Generativa tiene hacia la automatización y personalización de servicios, permitiendo la
    eficiencia y mejora operativa del negocio, así como la generación de nuevas fuentes de
    ingreso.

    Este proyecto tiene con objetivo el crear un "Banco de caso de usos de inteligencia
    Artificial Generativa en distintas industrias". En el cual se busca resolver un problema
    común en la industria que se pueda aromatizar.

    Dentro del proyecto veremos el caso de uso, en el cual una empresa llamada genera
    reportes de la base de datos de manera manual el cual hace el proceso ineficiente y
    costoso en horas humanas. Buscaremos automatizar el proceso de visualización o
    generación de reportes conectados con una base de datos de manera automatizada
    utilizando Inteligencia Artificial Generativa. 

    #### 2.1 Formulación del problema: ¿Qué es lo que se intenta resolver?
    Actualmente, la industria de tecnologia se enfrenta un proceso manual para generar
    reportes utilizando consultas directas a la base de datos y herramientas como Crystal
    Reports. 

    #### 2.2 Contexto: ¿Por qué es importante resolver este problema?
    Resolver este problema es crucial porque:
    1. Eficiencia operativa: La generación manual de reportes consume tiempo que
    podría destinarse a actividades estratégicas.
    2. Precisión: El proceso manual tiene margen de error, lo que puede afectar la toma
    de decisiones.
    3. Escalabilidad: Con el crecimiento de la empresa, el volumen de datos y la
    demanda de reportes aumentarán, haciendo insostenible el enfoque actual.
    4. Competitividad: Automatizar este proceso con IA generativa permitiría a la
    empresa mantenerse a la vanguardia en el uso de tecnologías modernas,
    mejorando su capacidad de análisis y respuesta.
    '''
    st.markdown(multi)

# Tab 2: Generate Report (Main functionality)
with tab2:
    index_sql = initialize_pinecone_index("track-rag-sql")
    rag = RAGBigQuerySystem(index_sql)
    st.header("Generar Reporte")
    user_input = st.text_input("Escribe el reporte que deseas generar:", 
                              placeholder="e.g. Generar reporte de dispositivos en violacion")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        generate_button = st.button("Generate Report", use_container_width=True)
    
    if generate_button and user_input:
        with st.spinner("Generando SQL y ejecutando consulta..."):
            sql_query = rag.generate_sql_query(user_input)
            
            st.subheader("SQL Query")
            st.code(sql_query, language="sql")
            
            # Fetch Data
            try:
                with st.spinner("Ejecutando consulta en BigQuery..."):
                    result_df = rag.query_bigquery(sql_query)
                
                st.subheader("Resultados")
                st.dataframe(result_df, use_container_width=True)
                
                # Check if dataframe has required lat/long columns
                required_cols = ['Latitude', 'Longitude']
                alt_required_cols = ['lat', 'lng']
                template = "map_template.html"
                
                # Display map if coordinates are available
                if all(col in result_df.columns for col in required_cols):
                    st.subheader("Visualización en Mapa")
                    map_data = result_df[required_cols].rename(columns={'Latitude': 'lat', 'Longitude': 'lng'})
                    map_data_for_display = map_data.rename(columns={'lng': 'lon'})
                    st.map(map_data_for_display)
                elif all(col in result_df.columns for col in alt_required_cols):
                    st.subheader("Visualización en Mapa")
                    map_data = result_df[alt_required_cols]
                    map_data_for_display = map_data.rename(columns={'lng': 'lon'})
                    st.map(map_data_for_display)
                else:
                    template = "table_template.html"
                    map_data = None
                
                # Generate PDF
                template_path = os.path.join(os.getcwd(), "src", "templates", template)
                
                with st.spinner("Generando reporte PDF..."):
                    pdf_path = generate_pdf_from_df(
                        dataFrame=result_df,
                        template_path=template_path,
                        output_filename="results/report.pdf",
                        template_type="table"
                    )
                
                st.success("Reporte generado exitosamente!")
                
                # PDF download and preview
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.download_button(
                        label="⬇️ Descargar Reporte PDF", 
                        data=open(pdf_path, "rb"), 
                        file_name="report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                st.subheader("Vista previa del reporte:")
                # Read the PDF file into memory
                with open(pdf_path, "rb") as file:
                    pdf_bytes = file.read()
                    st.markdown(f"""
                        <embed src="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"
                            width="100%" height="500" type="application/pdf">
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating report: {e}")

# Tab 3: Documentation
with tab3:
    st.header("Documentación")
    st.subheader("Instrucciones de uso")
    st.write("""
    1. Ve a la pestaña "Generar Reporte"
    2. Escribe en lenguaje natural lo que deseas consultar, por ejemplo:
        - "Mostrar dispositivos IoT con alertas críticas"
        - "Listar dispositivos ubicados en la Ciudad de México"
        - "Dispositivos con temperatura mayor a 80 grados en el último mes"
    3. Haz clic en "Generate Report" para ejecutar la consulta
    4. Visualiza los resultados y descarga el PDF si lo necesitas
    """)
    
    st.subheader("Ejemplos de consultas")
    examples = [
        "Generar reporte de dispositivos en violacion",
        "Mostrar dispositivos con bajo nivel de batería",
        "Listar todas las ubicaciones de dispositivos en CDMX",
        "Dispositivos que no se han conectado en la última semana"
    ]
    
    for example in examples:
        if st.button(example, key=f"example_{example}"):
            # This will set the query text in the input field in the other tab
            st.session_state['query_text'] = example
            st.rerun()
