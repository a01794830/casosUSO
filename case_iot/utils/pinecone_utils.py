"""
Utilidades para operaciones de base de datos vectorial Pinecone.
Optimizado para el caso de uso IoT.
"""
import os
import time
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from case_iot.utils.embedding_utils import get_embedding_new
from config import Config

logger = logging.getLogger(__name__)

# Configuración de Pinecone
PINECONE_API_KEY = Config.PINECONE_API_KEY
PINECONE_REGION = Config.PINECONE_REGION
PINECONE_CLOUD = Config.PINECONE_CLOUD
INDEX_NAME = Config.INDEX_NAME_IOT
VECTOR_DIM = Config.VECTOR_DIM

# Inicializar cliente de Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_or_create_index():
    """
    Obtiene o crea un índice Pinecone para el sistema IoT.
    
    Returns:
        Any: Objeto índice Pinecone inicializado
        
    Raises:
        RuntimeError: Si no se puede crear o conectar al índice
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
            wait_time = 0
            max_wait = 60  # Segundos máximos de espera
            
            while wait_time < max_wait:
                status = pc.describe_index(INDEX_NAME).status
                if status.get("ready", False):
                    break
                time.sleep(1)
                wait_time += 1
                
            if wait_time >= max_wait:
                logger.warning(f"Timeout esperando inicialización del índice {INDEX_NAME}")
            
            logger.info(f"Índice '{INDEX_NAME}' creado correctamente")
        else:
            logger.info(f"Conectando a índice existente: '{INDEX_NAME}'")
        
        # Conectar al índice
        index = pc.Index(INDEX_NAME)
        return index
        
    except Exception as e:
        logger.error(f"Error al crear/conectar al índice Pinecone: {e}", exc_info=True)
        raise RuntimeError(f"No se pudo inicializar el índice Pinecone: {str(e)}")

def interpret_and_search(
    user_query: str, 
    top_k: int = 2000, 
    re_rank_top: int = 200
) -> Tuple[List[str], bool, bool]:
    """
    Interpreta una consulta y busca documentos relevantes usando filtrado o embeddings.
    
    Primero intenta interpretar la consulta para extraer filtros (como device_id, 
    battery_level, etc.) y aplicarlos. Si esto falla, realiza una búsqueda por 
    similitud de embeddings.
    
    Args:
        user_query (str): Consulta del usuario en lenguaje natural
        top_k (int): Número máximo de documentos a recuperar
        re_rank_top (int): Número máximo de documentos para re-ranking
        
    Returns:
        Tuple[List[str], bool, bool]: 
            - Lista de textos relevantes
            - Flag indicando si se usó filtrado (True) o no (False)
            - Flag indicando si se usó fallback (True) o no (False)
    """
    if not user_query or not user_query.strip():
        logger.warning("Consulta vacía proporcionada")
        return [], False, False
        
    logger.info(f"Procesando consulta: '{user_query[:50]}...'")
    
    try:
        # 1. Intentar extraer filtros de la consulta usando LLM
        filter_json_str = call_llm_to_get_filterJSON(user_query)
        logger.info(f"JSON de filtros generado: {filter_json_str}")
        
        # 2. Intentar parsear el JSON
        try:
            filter_dict = json.loads(filter_json_str)
            
            # Si el JSON es válido pero vacío, ir directamente a búsqueda por embedding
            if not filter_dict:
                logger.info("JSON de filtros vacío, usando búsqueda por embedding")
                return embedding_search(user_query, top_k, re_rank_top), False, True
                
            # 3. Aplicar filtros si son válidos
            docs = apply_filter(filter_dict, top_k)
            
            if docs:
                # Re-ranking de documentos para mejorar relevancia
                final_contexts = re_rank_in_batches(docs, user_query, re_rank_top)
                logger.info(f"Búsqueda por filtros exitosa: {len(final_contexts)} documentos relevantes")
                return final_contexts, True, False
            else:
                # Si no hay resultados con filtros, usar fallback
                logger.info("No hay resultados con filtros, usando fallback")
                return embedding_search(user_query, top_k, re_rank_top), False, True
                
        except json.JSONDecodeError as e:
            # Error al parsear JSON, usar fallback
            logger.warning(f"Error al parsear JSON de filtros: {e}")
            return embedding_search(user_query, top_k, re_rank_top), False, True
            
    except Exception as e:
        # Error general, usar fallback
        logger.error(f"Error en interpret_and_search: {e}", exc_info=True)
        return embedding_search(user_query, top_k, re_rank_top), False, True

def call_llm_to_get_filterJSON(
    query: str, 
    model: str = "gpt-4o-mini",
    max_retries: int = 2
) -> str:
    """
    Usa OpenAI para "interpretar" la consulta y extraer filtros en formato JSON.
    
    Args:
        query (str): Consulta del usuario
        model (str): Modelo de OpenAI a utilizar
        max_retries (int): Número máximo de reintentos
        
    Returns:
        str: String JSON con filtros extraídos
    """
    system_prompt = """
Eres un parser especializado en extraer filtros de consultas sobre dispositivos IoT.
Dada la pregunta del usuario, devuelve un JSON con los filtros que puedas inferir.

Formato del JSON:
{
  "device_id": {"$eq": "abc123"},
  "battery_level": {"$lt": 10},
  "status": {"$eq": 1},
  "user_id": {"$eq": "xxyyzz"},
  "tamper_detected": {"$eq": true}
}

Operadores soportados:
- $eq: Igual
- $lt: Menor que
- $gt: Mayor que
- $lte: Menor o igual que
- $gte: Mayor o igual que

Si no se puede detectar ningún filtro, responde con un JSON vacío: {}
Responde ÚNICAMENTE con el JSON, sin texto adicional.
"""
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # Implementación con reintentos
    for retry in range(max_retries + 1):
        try:
            logger.debug(f"Generando filtros JSON para query: '{query[:50]}...'")
            
            # Llamar a la API de OpenAI
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,  # Respuestas determinísticas
                max_tokens=200
            )
            
            # Extraer y limpiar respuesta
            raw_txt = completion.choices[0].message.content.strip()
            logger.debug(f"Filtros JSON generados: {raw_txt}")
            return raw_txt
            
        except Exception as e:
            if retry < max_retries:
                # Esperar con backoff exponencial
                wait_time = 2 ** retry
                logger.warning(f"Error al generar filtros JSON (intento {retry+1}/{max_retries}): {str(e)}. Reintentando en {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Error persistente, devolver JSON vacío
                logger.error(f"Error persistente al generar filtros JSON: {e}", exc_info=True)
                return "{}"

def apply_filter(
    filter_dict: Dict[str, Any], 
    top_k: int = 2000
) -> List[str]:
    """
    Aplica filtros en Pinecone y devuelve documentos filtrados.
    
    Args:
        filter_dict (Dict[str, Any]): Diccionario de filtros
        top_k (int): Número máximo de documentos a recuperar
        
    Returns:
        List[str]: Lista de textos de documentos filtrados
    """
    if not filter_dict:
        logger.warning("Diccionario de filtros vacío")
        return []
        
    logger.info(f"Aplicando filtros: {json.dumps(filter_dict, ensure_ascii=False)}")
    
    try:
        # Obtener índice
        index = get_or_create_index()
        
        # Vector dummy para consulta
        dummy_vec = [0.0] * VECTOR_DIM
        
        # Realizar consulta con vector dummy (para recuperar todo)
        res = index.query(
            vector=dummy_vec,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        
        if not res or not hasattr(res, 'matches') or not res.matches:
            logger.info("No se encontraron documentos en el índice")
            return []
            
        # Aplicar filtros localmente
        matched_docs = []
        
        for match in res.matches:
            metadata = match.metadata
            
            # Verificar si el documento pasa todos los filtros
            if passes_filter(metadata, filter_dict):
                text = metadata.get("TEXT", "")
                if text:
                    matched_docs.append(text)
        
        logger.info(f"Filtrado local completado: {len(matched_docs)} documentos coincidentes")
        return matched_docs
        
    except Exception as e:
        logger.error(f"Error al aplicar filtros: {e}", exc_info=True)
        return []

def passes_filter(
    md: Dict[str, Any], 
    filter_dict: Dict[str, Dict[str, Any]]
) -> bool:
    """
    Verifica si un documento pasa todos los criterios de filtro.
    
    Args:
        md (Dict[str, Any]): Metadatos del documento
        filter_dict (Dict[str, Dict[str, Any]]): Diccionario de filtros
        
    Returns:
        bool: True si pasa todos los filtros, False en caso contrario
    """
    for field, condition in filter_dict.items():
        # Cada condition es un diccionario {"$eq": x} o {"$lt": x}, etc.
        
        # Verificar operador $eq (igual)
        if "$eq" in condition:
            val = condition["$eq"]
            if not check_eq(md, field, val):
                return False
                
        # Verificar operador $lt (menor que)
        elif "$lt" in condition:
            val = condition["$lt"]
            if not check_lt(md, field, val):
                return False
                
        # Verificar operador $gt (mayor que)
        elif "$gt" in condition:
            val = condition["$gt"]
            if not check_gt(md, field, val):
                return False
                
        # Verificar operador $lte (menor o igual que)
        elif "$lte" in condition:
            val = condition["$lte"]
            if not check_lte(md, field, val):
                return False
                
        # Verificar operador $gte (mayor o igual que)
        elif "$gte" in condition:
            val = condition["$gte"]
            if not check_gte(md, field, val):
                return False
    
    # Si pasa todos los filtros
    return True

def check_eq(
    md: Dict[str, Any], 
    field: str, 
    val: Any
) -> bool:
    """
    Verifica si un campo es igual a un valor.
    
    Args:
        md (Dict[str, Any]): Metadatos del documento
        field (str): Campo a verificar
        val (Any): Valor esperado
        
    Returns:
        bool: True si el campo es igual al valor, False en caso contrario
    """
    # Obtener valor del campo
    meta_val_str = md.get(field)
    
    # Si el campo no existe, no pasa el filtro
    if meta_val_str is None:
        return False
        
    # Convertir meta_val_str a string si no lo es
    if not isinstance(meta_val_str, str):
        meta_val_str = str(meta_val_str)
    
    # Manejar valores booleanos
    if isinstance(val, bool):
        return meta_val_str.lower() == str(val).lower()
    
    # Manejar valores numéricos
    if isinstance(val, (int, float)):
        try:
            num = float(meta_val_str)
            return num == val
        except ValueError:
            return False
    
    # Manejar strings (comparación directa)
    return meta_val_str == val

def check_lt(
    md: Dict[str, Any], 
    field: str, 
    val: Union[int, float]
) -> bool:
    """
    Verifica si un campo es menor que un valor.
    
    Args:
        md (Dict[str, Any]): Metadatos del documento
        field (str): Campo a verificar
        val (Union[int, float]): Valor de comparación
        
    Returns:
        bool: True si el campo es menor que el valor, False en caso contrario
    """
    meta_val_str = md.get(field)
    
    if meta_val_str is None:
        return False
        
    try:
        num = float(meta_val_str)
        return num < val
    except (ValueError, TypeError):
        return False

def check_gt(
    md: Dict[str, Any], 
    field: str, 
    val: Union[int, float]
) -> bool:
    """
    Verifica si un campo es mayor que un valor.
    
    Args:
        md (Dict[str, Any]): Metadatos del documento
        field (str): Campo a verificar
        val (Union[int, float]): Valor de comparación
        
    Returns:
        bool: True si el campo es mayor que el valor, False en caso contrario
    """
    meta_val_str = md.get(field)
    
    if meta_val_str is None:
        return False
        
    try:
        num = float(meta_val_str)
        return num > val
    except (ValueError, TypeError):
        return False

def check_lte(
    md: Dict[str, Any], 
    field: str, 
    val: Union[int, float]
) -> bool:
    """
    Verifica si un campo es menor o igual que un valor.
    
    Args:
        md (Dict[str, Any]): Metadatos del documento
        field (str): Campo a verificar
        val (Union[int, float]): Valor de comparación
        
    Returns:
        bool: True si el campo es menor o igual que el valor, False en caso contrario
    """
    meta_val_str = md.get(field)
    
    if meta_val_str is None:
        return False
        
    try:
        num = float(meta_val_str)
        return num <= val
    except (ValueError, TypeError):
        return False

def check_gte(
    md: Dict[str, Any], 
    field: str, 
    val: Union[int, float]
) -> bool:
    """
    Verifica si un campo es mayor o igual que un valor.
    
    Args:
        md (Dict[str, Any]): Metadatos del documento
        field (str): Campo a verificar
        val (Union[int, float]): Valor de comparación
        
    Returns:
        bool: True si el campo es mayor o igual que el valor, False en caso contrario
    """
    meta_val_str = md.get(field)
    
    if meta_val_str is None:
        return False
        
    try:
        num = float(meta_val_str)
        return num >= val
    except (ValueError, TypeError):
        return False

def embedding_search(
    query: str, 
    top_k: int = 2000, 
    re_rank_top: int = 200
) -> List[str]:
    """
    Realiza búsqueda por similitud de embeddings.
    
    Args:
        query (str): Consulta del usuario
        top_k (int): Número máximo de documentos a recuperar
        re_rank_top (int): Número máximo de documentos para re-ranking
        
    Returns:
        List[str]: Lista de textos de documentos relevantes
    """
    logger.info(f"Realizando búsqueda por embedding para: '{query[:50]}...'")
    
    try:
        # Obtener índice
        index = get_or_create_index()
        
        # Generar embedding para la consulta
        q_emb = get_embedding_new(query)
        
        # Realizar búsqueda
        res = index.query(
            vector=q_emb,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        
        if not hasattr(res, 'matches') or not res.matches:
            logger.info("No se encontraron documentos con embedding")
            return []
        
        # Extraer textos
        docs = []
        for match in res.matches:
            text = match.metadata.get("TEXT", "")
            if text:
                docs.append(text)
                
        # Re-ranking para mejorar relevancia
        final_docs = re_rank_in_batches(docs, query, re_rank_top)
        logger.info(f"Búsqueda por embedding completada: {len(final_docs)} documentos relevantes")
        return final_docs
        
    except Exception as e:
        logger.error(f"Error en búsqueda por embedding: {e}", exc_info=True)
        return []

def re_rank_in_batches(
    docs: List[str], 
    query: str, 
    top_n: int = 200
) -> List[str]:
    """
    Re-rankea documentos en lotes para manejar grandes volúmenes.
    
    Args:
        docs (List[str]): Lista de documentos a re-rankear
        query (str): Consulta del usuario
        top_n (int): Número final de documentos a retornar
        
    Returns:
        List[str]: Lista de documentos re-rankeados
    """
    if not docs:
        return []
        
    # Configurar tamaños de lote para evitar límites de API
    chunk_size = 100  # Máximo permitido por llamada
    partial_top = min(30, top_n)  # Resultados a mantener por lote
    
    logger.info(f"Re-ranking {len(docs)} documentos en lotes de {chunk_size}")
    
    # Lista para resultados parciales
    partial_res = []
    
    # Procesar documentos en lotes
    for i in range(0, len(docs), chunk_size):
        chunk = docs[i:i+chunk_size]
        chunk_reranked = re_rank_once(chunk, query, partial_top)
        partial_res.extend(chunk_reranked)
        logger.debug(f"Lote {i//chunk_size + 1} re-rankeado: {len(chunk_reranked)} resultados")
    
    # Si hay demasiados resultados parciales, reprocesar
    if len(partial_res) <= chunk_size:
        final_results = re_rank_once(partial_res, query, top_n)
        logger.info(f"Re-ranking completado: {len(final_results)} documentos finales")
        return final_results
    else:
        # Recursión para manejar grandes volúmenes
        logger.info(f"Re-procesando {len(partial_res)} resultados parciales")
        return re_rank_in_batches(partial_res, query, top_n)

def re_rank_once(
    docs: List[str], 
    query: str, 
    top_n: int
) -> List[str]:
    """
    Re-rankea documentos en una sola llamada.
    
    Args:
        docs (List[str]): Lista de documentos a re-rankear
        query (str): Consulta del usuario
        top_n (int): Número de documentos a retornar
        
    Returns:
        List[str]: Lista de documentos re-rankeados
    """
    if not docs:
        return []
        
    # Limitar top_n al número de documentos disponibles
    top_n = min(top_n, len(docs))
    
    try:
        # Usar reranker de Pinecone
        rr = pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=docs,
            top_n=top_n,
            return_documents=True
        )
        
        if not rr or not hasattr(rr, 'data') or not rr.data:
            logger.warning("No se obtuvieron resultados del re-ranking")
            return docs[:top_n]  # Fallback a los primeros N documentos
        
        # Extraer textos re-rankeados
        out = []
        for d in rr.data:
            if 'document' in d and 'text' in d['document']:
                out.append(d['document']['text'])
                
        logger.debug(f"Re-ranking exitoso: {len(out)} documentos")
        return out
        
    except Exception as e:
        logger.error(f"Error en re_rank_once: {e}", exc_info=True)
        # Fallback a los primeros N documentos
        return docs[:top_n]

def query_by_id(
    input_id: str, 
    tipo: str = "device_id"
) -> List[Dict[str, Any]]:
    """
    Filtra documentos por device_id o user_id.
    
    Args:
        input_id (str): Valor del ID a buscar
        tipo (str): Tipo de ID ("device_id" o "user_id")
        
    Returns:
        List[Dict[str, Any]]: Lista de metadatos de documentos coincidentes
        
    Raises:
        ValueError: Si el tipo no es válido
        RuntimeError: Si hay un error de conexión
    """
    # Validar tipo
    if tipo not in ["device_id", "user_id"]:
        raise ValueError(f"Tipo de ID no válido: {tipo}. Debe ser 'device_id' o 'user_id'")
        
    logger.info(f"Consultando por {tipo}={input_id}")
    
    try:
        # Obtener índice
        index = get_or_create_index()
        
        # Vector dummy para consulta
        dummy_vec = [0.0] * VECTOR_DIM
        
        # Construir filtro
        my_filter = {tipo: {"$eq": input_id}}
        
        # Realizar consulta
        res = index.query(
            vector=dummy_vec,
            top_k=5000,  # Valor alto para no perder resultados
            include_values=False,
            include_metadata=True,
            filter=my_filter
        )
        
        # Verificar resultados
        if res and hasattr(res, 'matches') and res.matches:
            # Extraer metadatos
            results = [match.metadata for match in res.matches]
            logger.info(f"Se encontraron {len(results)} resultados para {tipo}={input_id}")
            return results
            
        logger.info(f"No se encontraron resultados para {tipo}={input_id}")
        return []
        
    except Exception as e:
        logger.error(f"Error en query_by_id: {e}", exc_info=True)
        raise RuntimeError(f"Error consultando por ID: {str(e)}")
