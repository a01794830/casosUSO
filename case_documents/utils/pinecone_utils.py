"""
Utilidades para operaciones con Pinecone en el sistema de análisis de documentos.
"""
import os
import time
import logging
import uuid
from typing import List, Dict, Any, Optional, Union
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from case_documents.utils.embedding_utils import get_embedding_new
from config import Config

logger = logging.getLogger(__name__)
load_dotenv()

# Configuración de Pinecone
PINECONE_API_KEY = Config.PINECONE_API_KEY
PINECONE_REGION = Config.PINECONE_REGION
PINECONE_CLOUD = Config.PINECONE_CLOUD
INDEX_NAME = Config.INDEX_NAME_DOCS
VECTOR_DIM = Config.VECTOR_DIM

# Inicializar cliente Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_or_create_index() -> Any:
    """
    Obtiene o crea un índice Pinecone para el sistema de documentos.
    
    Returns:
        Any: Objeto índice Pinecone inicializado
    """
    logger.info(f"Obteniendo o creando índice Pinecone: {INDEX_NAME}")
    
    try:
        # Verificar si el índice ya existe
        existing_indexes = pc.list_indexes().names()
        
        if INDEX_NAME not in existing_indexes:
            logger.info(f"Creando nuevo índice: {INDEX_NAME}")
            
            # Crear índice con especificaciones adecuadas
            pc.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            
            # Esperar a que el índice esté listo
            ready = False
            max_wait = 60  # Segundos máximos de espera
            start_time = time.time()
            
            while not ready and time.time() - start_time < max_wait:
                try:
                    status = pc.describe_index(INDEX_NAME).status
                    ready = status.get("ready", False)
                    if ready:
                        break
                    time.sleep(2)
                except Exception as e:
                    logger.warning(f"Error al verificar estado del índice: {e}")
                    time.sleep(2)
            
            if not ready:
                logger.warning(f"Timeout esperando que el índice esté listo, continuando...")
                
            logger.info(f"Índice '{INDEX_NAME}' creado")
        else:
            logger.info(f"Usando índice existente: '{INDEX_NAME}'")
        
        # Conectar al índice
        index = pc.Index(INDEX_NAME)
        return index
        
    except Exception as e:
        logger.error(f"Error al inicializar índice Pinecone: {e}", exc_info=True)
        raise RuntimeError(f"No se pudo inicializar el índice Pinecone: {str(e)}")

def upsert_docs(chunks: List[str], batch_size: int = 50) -> bool:
    """
    Inserta fragmentos de texto en Pinecone con embeddings.
    
    Args:
        chunks (List[str]): Lista de fragmentos de texto a insertar
        batch_size (int): Tamaño del lote para operaciones de upsert
        
    Returns:
        bool: True si la operación fue exitosa
    """
    if not chunks:
        logger.warning("No hay fragmentos para insertar")
        return False
        
    logger.info(f"Insertando {len(chunks)} fragmentos en Pinecone")
    
    try:
        # Obtener o crear índice
        index = get_or_create_index()
        
        # Procesar en lotes
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            logger.debug(f"Procesando lote {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} fragmentos)")
            
            # Generar embeddings para el lote
            vectors = []
            for chunk in batch:
                # Generar embedding
                emb = get_embedding_new(chunk)
                
                # Crear identificador único
                doc_id = str(uuid.uuid4())
                
                # Crear metadata
                meta = {"TEXT": chunk}
                
                # Añadir a la lista de vectores
                vectors.append((doc_id, emb, meta))
            
            # Insertar lote en Pinecone
            index.upsert(vectors=vectors)
            logger.debug(f"Lote de {len(vectors)} vectores insertado correctamente")
        
        logger.info(f"Todos los fragmentos ({len(chunks)}) insertados correctamente")
        return True
        
    except Exception as e:
        logger.error(f"Error al insertar fragmentos en Pinecone: {e}", exc_info=True)
        return False

def search_docs(query: str, top_k: int = 100) -> List[str]:
    """
    Busca documentos similares a una consulta.
    
    Args:
        query (str): Consulta del usuario
        top_k (int): Número máximo de resultados a retornar
        
    Returns:
        List[str]: Lista de textos de documentos relevantes
    """
    if not query.strip():
        logger.warning("Consulta vacía")
        return []
        
    logger.info(f"Buscando documentos para consulta: '{query[:50]}...'")
    
    try:
        # Generar embedding para la consulta
        q_emb = get_embedding_new(query)
        
        # Obtener índice
        index = get_or_create_index()
        
        # Realizar búsqueda
        res = index.query(
            vector=q_emb,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        
        # Verificar resultados
        if not res or not res.matches:
            logger.info("No se encontraron resultados para la consulta")
            return []
        
        # Extraer textos de los resultados
        docs = []
        for match in res.matches:
            txt = match.metadata.get("TEXT", "")
            if txt:
                docs.append(txt)
        
        logger.info(f"Se encontraron {len(docs)} documentos relevantes")
        return docs
        
    except Exception as e:
        logger.error(f"Error al buscar documentos: {e}", exc_info=True)
        return []

def get_all_docs(top_k: int = 5000) -> List[str]:
    """
    Descarga todos los documentos almacenados en el índice.
    
    Args:
        top_k (int): Número máximo de documentos a recuperar
        
    Returns:
        List[str]: Lista de textos de todos los documentos
    """
    logger.info(f"Obteniendo todos los documentos (hasta {top_k})")
    
    try:
        # Obtener índice
        index = get_or_create_index()
        
        # Vector dummy para recuperar todos los documentos
        dummy_vec = [0.0] * VECTOR_DIM
        
        # Consulta para recuperar todo
        res = index.query(
            vector=dummy_vec,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        
        # Verificar resultados
        if not res or not res.matches:
            logger.info("No se encontraron documentos en el índice")
            return []
        
        docs = []
        for match in res.matches:
            txt = match.metadata.get("TEXT", "")
            if txt:
                docs.append(txt)
        
        logger.info(f"Se recuperaron {len(docs)} documentos en total")
        return docs
        
    except Exception as e:
        logger.error(f"Error al recuperar todos los documentos: {e}", exc_info=True)
        return []
