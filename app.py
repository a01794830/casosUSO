import streamlit as st
import logging
from config import Config
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("genai-app")

# Configuración de la página
st.set_page_config(
    page_title="GenAI Demo App", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Verificar claves API al inicio
if not Config.OPENAI_API_KEY:
    st.error("⚠️ No se ha configurado OPENAI_API_KEY. Por favor configura las variables de entorno.")
elif not Config.PINECONE_API_KEY:
    st.warning("⚠️ No se ha configurado PINECONE_API_KEY. Algunas funcionalidades pueden no estar disponibles.")

# Navegación principal
pg = st.navigation([
    st.Page("pages/page_home.py", title="Inicio", icon=":material/home:"), 
    st.Page("case_bigquery_sql/page_case_1.py", title="Caso 1 - SQL", icon=":material/database:"),
    st.Page("case_iot/page_case_2.py", title="Caso 2 - IoT", icon=":material/devices:"),
    st.Page("case_edu/page_case_3.py", title="Caso 3 - Educación", icon=":material/school:"),
    st.Page("case_documents/case_documents.py", title="Caso 4 - Documentos", icon=":material/file-document:")
])

pg.run()
