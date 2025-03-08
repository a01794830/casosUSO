"""
Utilidades para la generación de embeddings y respuestas usando OpenAI.
Optimizado para el caso de uso IoT.
"""
import os
import logging
import time
from typing import List, Any, Optional, Union
from openai import OpenAI
from config import Config

logger = logging.getLogger(__name__)

# Crear cliente OpenAI una vez
client = OpenAI(api_key=Config.OPENAI_API_KEY)

def get_embedding_new(
    text: str, 
    model: str = "text-embedding-3-small",
    max_retries: int = 3
) -> List[float]:
    """
    Genera embedding vectorial para texto usando la API de OpenAI.
    
    Args:
        text (str): Texto para generar el embedding
        model (str): Modelo de embeddings a utilizar
        max_retries (int): Número máximo de reintentos
        
    Returns:
        List[float]: Vector de embedding generado
        
    Raises:
        ValueError: Si el texto está vacío
        RuntimeError: Si hay un error persistente con la API
    """
    # Validar entrada
    if not text or not text.strip():
        logger.error("Texto vacío proporcionado para embedding")
        raise ValueError("El texto no puede estar vacío")
    
    # Preprocesar texto
    text = text.strip()
    
    # Implementación con reintentos
    for retry in range(max_retries + 1):
        try:
            logger.debug(f"Generando embedding para texto de {len(text)} caracteres con modelo {model}")
            
            # Llamar a la API de OpenAI
            response = client.embeddings.create(
                model=model,
                input=text,
                encoding_format="float"
            )
            
            # Extraer y devolver el embedding
            embedding = response.data[0].embedding
            logger.debug(f"Embedding generado correctamente: {len(embedding)} dimensiones")
            return embedding
            
        except Exception as e:
            if retry < max_retries:
                # Esperar con backoff exponencial
                wait_time = 2 ** retry
                logger.warning(f"Error al generar embedding (intento {retry+1}/{max_retries}): {str(e)}. Reintentando en {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Error persistente
                logger.error(f"Error persistente generando embedding: {str(e)}")
                raise RuntimeError(f"Error al generar embedding después de {max_retries} intentos: {str(e)}")

def generate_chat_response(
    contexts: List[str], 
    user_query: str, 
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 500,
    max_retries: int = 2
) -> str:
    """
    Genera una respuesta para una consulta sobre dispositivos IoT
    basada en contextos relevantes.
    
    Args:
        contexts (List[str]): Lista de contextos relevantes
        user_query (str): Consulta del usuario
        model (str): Modelo de OpenAI a utilizar
        max_tokens (int): Longitud máxima de la respuesta
        max_retries (int): Número máximo de reintentos
        
    Returns:
        str: Respuesta generada
    """
    # Validar entradas
    if not contexts:
        logger.warning("No se proporcionaron contextos para generar respuesta")
        return "No tengo suficiente información sobre los dispositivos IoT para responder a tu consulta."
    
    if not user_query or not user_query.strip():
        logger.warning("Consulta vacía proporcionada")
        return "Por favor, especifica tu consulta sobre los dispositivos IoT."
    
    # Limitar el número de contextos para evitar tokens excesivos
    if len(contexts) > 5:
        logger.info(f"Limitando contextos de {len(contexts)} a 5")
        contexts = contexts[:5]
    
    # Unir contextos con separadores claros
    joined_context = "\n\n---\n\n".join(contexts)
    
    # Construir prompt
    prompt = f"""Usa el siguiente contexto sobre dispositivos IoT para responder la consulta del usuario:
    
{joined_context}

---

CONSULTA: {user_query}

Responde de manera concisa y directa. Si la información necesaria no está en el contexto,
indica claramente: "No tengo esa información sobre los dispositivos IoT."

Si se trata de ubicaciones geográficas, proporciona las coordenadas exactas cuando estén disponibles.
Si se trata de alertas o problemas, indica la severidad y cuándo fueron reportados.
"""
    
    # Implementación con reintentos
    for retry in range(max_retries + 1):
        try:
            logger.debug(f"Generando respuesta para query: '{user_query[:50]}...'")
            
            # Llamar a la API de OpenAI
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Eres un asistente especializado en monitoreo de dispositivos IoT que proporciona información precisa y técnica."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Respuestas determinísticas
                max_tokens=max_tokens
            )
            
            # Extraer y devolver la respuesta
            answer = response.choices[0].message.content
            logger.info("Respuesta generada correctamente")
            return answer
            
        except Exception as e:
            if retry < max_retries:
                # Esperar con backoff exponencial
                wait_time = 2 ** retry
                logger.warning(f"Error al generar respuesta (intento {retry+1}/{max_retries}): {str(e)}. Reintentando en {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Error persistente
                logger.error(f"Error persistente generando respuesta: {str(e)}")
                return f"Lo siento, no pude generar una respuesta debido a un error técnico. Por favor, intenta nuevamente. Error: {str(e)}"
