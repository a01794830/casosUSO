"""
Servicio de generación de embeddings usando OpenAI.
Proporciona utilidades para la vectorización de texto.
"""
import logging
from typing import List, Union
from openai import OpenAI
from config import Config

logger = logging.getLogger(__name__)

# Inicializar cliente de OpenAI con configuración global
client = OpenAI(api_key=Config.OPENAI_API_KEY)

def get_openai_embeddings(
    texts: Union[str, List[str]], 
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """
    Genera embeddings para uno o varios textos utilizando la API de OpenAI.
    
    Args:
        texts (Union[str, List[str]]): Texto único o lista de textos a procesar
        model (str, optional): Modelo de embeddings a utilizar. Default: "text-embedding-3-small"
    
    Returns:
        List[List[float]]: Lista de vectores de embeddings generados
        
    Raises:
        ValueError: Si los textos están vacíos
        Exception: Si ocurre un error en la API
    """
    # Asegurar que texts sea una lista
    if isinstance(texts, str):
        texts = [texts]
        
    if not texts:
        logger.error("La lista de textos está vacía")
        raise ValueError("La lista de textos no puede estar vacía")
    
    # Pre-procesamiento de textos
    processed_texts = [text.replace("\n", " ").strip() for text in texts]
    processed_texts = [text for text in processed_texts if text]  # Eliminar vacíos
    
    if not processed_texts:
        logger.error("Todos los textos quedaron vacíos después del procesamiento")
        raise ValueError("Todos los textos quedaron vacíos después del procesamiento")
    
    try:
        logger.debug(f"Generando embeddings para {len(processed_texts)} textos con modelo {model}")
        response = client.embeddings.create(
            model=model,
            input=processed_texts,
            encoding_format="float"
        )
        
        # Extraer y retornar embeddings
        embeddings = [item.embedding for item in response.data]
        logger.debug(f"Embeddings generados correctamente: {len(embeddings)} vectores")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error al generar embeddings: {str(e)}")
        raise Exception(f"Error al generar embeddings: {str(e)}")
