"""
Utilidades para generar embeddings y respuestas usando OpenAI.
"""
import os
import logging
import time
from typing import List, Optional, Dict, Any, Union
from openai import OpenAI
from config import Config

logger = logging.getLogger(__name__)

# Inicializar cliente una vez
client = OpenAI(api_key=Config.OPENAI_API_KEY)

def get_embedding_new(
    text: Union[str, List[str]], 
    model: str = "text-embedding-3-small",
    max_retries: int = 3
) -> Union[List[float], List[List[float]]]:
    """
    Genera embeddings usando OpenAI API (compatible con openai>=1.0.0).
    
    Args:
        text (Union[str, List[str]]): Texto o lista de textos para embeddings
        model (str): Modelo de embeddings a utilizar
        max_retries (int): Número máximo de reintentos en caso de error
        
    Returns:
        Union[List[float], List[List[float]]]: Embeddings generados
        
    Raises:
        ValueError: Si el texto está vacío
        RuntimeError: Si hay un error persistente con la API
    """
    # Validar entrada
    if isinstance(text, str):
        if not text.strip():
            logger.error("Texto vacío proporcionado para embedding")
            raise ValueError("El texto no puede estar vacío")
        input_text = [text.strip()]
        single_input = True
    else:
        if not text or all(not t.strip() for t in text):
            logger.error("Lista de textos vacía proporcionada para embedding")
            raise ValueError("La lista de textos no puede estar vacía")
        input_text = [t.strip() for t in text if t.strip()]
        single_input = False
    
    # Intentar con reintentos en caso de error
    retries = 0
    while retries <= max_retries:
        try:
            logger.debug(f"Generando embeddings para {len(input_text)} textos con modelo {model}")
            
            response = client.embeddings.create(
                model=model,
                input=input_text,
                encoding_format="float"
            )
            
            # Extraer embeddings
            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Embeddings generados correctamente: {len(embeddings)} vectores")
            
            # Devolver un solo embedding si la entrada era un solo texto
            if single_input:
                return embeddings[0]
            return embeddings
            
        except Exception as e:
            retries += 1
            logger.warning(f"Error al generar embeddings (intento {retries}/{max_retries}): {str(e)}")
            
            if retries <= max_retries:
                # Espera exponencial
                time.sleep(2 ** retries)
            else:
                logger.error(f"Error persistente al generar embeddings: {str(e)}")
                # En caso de fallo total, devolver un vector de ceros como fallback
                if single_input:
                    logger.warning("Devolviendo vector de ceros como fallback")
                    return [0.0] * Config.VECTOR_DIM
                return [[0.0] * Config.VECTOR_DIM for _ in range(len(input_text))]

def generate_chat_response(
    contexts: List[str], 
    query: str, 
    model: str = "gpt-3.5-turbo",
    max_retries: int = 2
) -> str:
    """
    Genera una respuesta basada en contextos y una consulta utilizando ChatGPT.
    
    Args:
        contexts (List[str]): Lista de contextos relevantes
        query (str): Consulta del usuario
        model (str): Modelo de OpenAI a utilizar
        max_retries (int): Número máximo de reintentos
        
    Returns:
        str: Respuesta generada
    """
    # Unir contextos (limitando a los primeros 3 para evitar token overflow)
    limit_contexts = contexts[:3] if len(contexts) > 3 else contexts
    joined_context = "\n\n---\n\n".join(limit_contexts)
    
    # Construir prompt
    prompt = f"""Basado en este texto:
    
{joined_context}

Pregunta: {query}

Si no encuentras la información necesaria para responder, di: "No encuentro esa información en los documentos proporcionados."
Responde de forma clara y concisa.
"""
    
    # Reintentos en caso de error
    retries = 0
    while retries <= max_retries:
        try:
            logger.debug(f"Generando respuesta para query: {query[:50]}...")
            
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Eres un asistente especializado en analizar documentos subidos por el usuario."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            answer = completion.choices[0].message.content
            logger.info("Respuesta generada correctamente")
            return answer
            
        except Exception as e:
            retries += 1
            logger.warning(f"Error al generar respuesta (intento {retries}/{max_retries}): {str(e)}")
            
            if retries <= max_retries:
                time.sleep(2 ** retries)
            else:
                logger.error(f"Error persistente al generar respuesta: {str(e)}")
                return f"Lo siento, no pude generar una respuesta debido a un error técnico. Por favor, intenta nuevamente más tarde. (Error: {str(e)})"
