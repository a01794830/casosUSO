import streamlit as st
import pandas as pd
import numpy as np

# Set up the page configuration
st.set_page_config(page_title="GenAI App", page_icon="📊", layout="wide")
pg = st.navigation([st.Page("pages/page_home.py",title="Inicio",icon=":material/home:"), 
                    st.Page("case_bigquery_sql/page_case_1.py",title="Caso 1 - SQL",icon=":material/inventory:"),
                    st.Page("case_iot/page_case_2.py",title="Caso 2 - IoT",icon=":material/inventory:"),
                st.Page("case_edu/page_case_3.py",title="Caso 3 - Educación",icon=":material/inventory:"),
                    st.Page("case_documents/case_documents.py",title="Caso 4 - Documentos",icon=":material/inventory:")])
pg.run()