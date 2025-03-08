import os
import logging
from dotenv import load_dotenv
import streamlit as st

logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class Config:
    """
    Clase de configuración centralizada para la aplicación.
    Prioriza las variables de Streamlit secrets sobre variables de entorno.
    """
    
    # Intentar cargar desde Streamlit secrets (producción)
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
        PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
        PINECONE_ENV = st.secrets.get("PINECONE_ENV", "us-west1-gcp")
        BIGQUERY_PROJECT_ID = st.secrets.get("PROJECT_ID")
    except:
        # Fallback a variables de entorno (desarrollo)
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
        BIGQUERY_PROJECT_ID = os.getenv("PROJECT_ID")
    
    # Configuración de Pinecone
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    
    # Índices de Pinecone
    INDEX_NAME_SQL = "track-rag-sql"
    INDEX_NAME_IOT = "tracking1-rag"
    INDEX_NAME_DOCS = "doc-rag"
    
    # Dimensiones de vectores
    VECTOR_DIM = 1536
    
    # Configuración de BigQuery
    DATASET_ID = "tracking_dataset"
    TABLE_ID = "tracking_data"
    
    @classmethod
    def validate_config(cls):
        """Valida que las configuraciones esenciales existan."""
        missing = []
        
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if not cls.BIGQUERY_PROJECT_ID:
            missing.append("PROJECT_ID")
            
        if missing:
            logger.warning(f"Faltan las siguientes configuraciones: {', '.join(missing)}")
            return False
        return True
