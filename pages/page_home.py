import streamlit as st

st.title("Banco de Casos de Uso GenAI")
multi = '''
Equipo 16

De acuerdo con las publicaciones de Gartner [1], el potencial que la Inteligencia Artificial
Generativa tiene hacia la automatización y personalización de servicios, permitiendo la
eficiencia y mejora operativa del negocio, así como la generación de nuevas fuentes de
ingreso.

Bacos de caso de uso:
'''
st.markdown(multi)
st.page_link("case_bigquery_sql/page_case_1.py", label="Ver Caso 1")
st.page_link("case_iot/page_case_2.py", label="Ver Caso 2")
st.page_link("case_edu/page_case_3.py", label="Ver Caso 3")
st.page_link("case_documents/case_documents.py", label="Ver Caso 4")



#3 Gestion de Estudiantes 
#4 