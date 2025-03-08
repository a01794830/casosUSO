"""
Utilidades para generar resúmenes de documentos utilizando OpenAI.
"""
import logging
import time
from typing import List, Optional
from openai import OpenAI
from config import Config

logger = logging.getLogger(__name__)

# Inicializar cliente OpenAI una vez
client = OpenAI(api_key=Config.OPENAI_API_KEY)

def summarize_global_docs(
    docs: List[str], 
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 500,
    max_retries: int = 2
) -> str:
    """
    Genera un resumen global de múltiples documentos o fragmentos.
    Maneja documentos grandes dividiendo el proceso en partes.
    
    Args:
        docs (List[str]): Lista de textos de documentos a resumir
        model (str): Modelo de OpenAI a utilizar
        max_tokens (int): Longitud máxima del resumen en tokens
        max_retries (int): Número máximo de reintentos
        
    Returns:
        str: Resumen global generado
    """
    logger.info(f"Generando resumen global para {len(docs)} documentos")
    
    if not docs:
        logger.warning("No hay documentos para resumir")
        return "No hay documentos indexados para resumir."
    
    # Si hay muchos documentos, tomar una muestra representativa
    MAX_DOCS = 20  # Límite de documentos a procesar
    if len(docs) > MAX_DOCS:
        logger.info(f"Limitando a {MAX_DOCS} documentos para el resumen global")
        
        # Tomar documentos distribuidos uniformemente
        step = len(docs) // MAX_DOCS
        sampled_docs = [docs[i] for i in range(0, len(docs), step)][:MAX_DOCS]
    else:
        sampled_docs = docs
    
    # Unir textos con separadores claros
    text_to_summarize = "\n\n---\n\n".join(sampled_docs)
    
    # Manejar textos muy grandes
    if len(text_to_summarize) > 20000:
        logger.info(f"Texto demasiado largo ({len(text_to_summarize)} caracteres), dividiendo en partes")
        return summarize_in_chunks(docs, model, max_tokens)
    
    # Construir prompt
    prompt = f"""Genera un resumen global coherente y completo de estos textos:

{text_to_summarize}

El resumen debe:
1. Enfatizar los puntos clave y conceptos principales
2. Mantener un tono objetivo y profesional
3. Estar estructurado de forma clara y lógica
4. Ser conciso pero informativo

Si no hay información suficiente, indícalo claramente.
"""
    
    # Intentar con reintentos en caso de error
    retries = 0
    while retries <= max_retries:
        try:
            logger.debug(f"Generando resumen con modelo {model}")
            
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en crear resúmenes concisos y precisos de documentos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            
            answer = completion.choices[0].message.content
            logger.info("Resumen global generado correctamente")
            return answer
            
        except Exception as e:
            retries += 1
            logger.warning(f"Error al generar resumen (intento {retries}/{max_retries}): {str(e)}")
            
            if retries <= max_retries:
                time.sleep(2 ** retries)
            else:
                logger.error(f"Error persistente al generar resumen global: {str(e)}", exc_info=True)
                return f"Error generando resumen global. Por favor, intenta nuevamente. Error: {str(e)}"

def summarize_in_chunks(
    docs: List[str], 
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 500
) -> str:
    """
    Genera resúmenes para documentos grandes dividiéndolos en partes.
    
    Args:
        docs (List[str]): Lista de documentos a resumir
        model (str): Modelo de OpenAI a utilizar
        max_tokens (int): Longitud máxima de cada resumen parcial
        
    Returns:
        str: Resumen global combinado
    """
    logger.info("Utilizando método de resumen por partes")
    
    # Determinar número de partes (chunks)
    CHUNK_SIZE = 5  # Documentos por chunk
    num_chunks = (len(docs) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    # Generar resúmenes parciales
    partial_summaries = []
    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, len(docs))
        
        logger.debug(f"Procesando chunk {i+1}/{num_chunks} (docs {start_idx+1}-{end_idx})")
        
        # Obtener documentos para este chunk
        chunk_docs = docs[start_idx:end_idx]
        chunk_text = "\n\n---\n\n".join(chunk_docs)
        
        # Generar resumen parcial
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Genera un resumen conciso de estos documentos."},
                    {"role": "user", "content": f"Resume este fragmento de texto:\n\n{chunk_text}"}
                ],
                temperature=0.3,
                max_tokens=max_tokens // 2
            )
            
            partial_summary = completion.choices[0].message.content
            partial_summaries.append(partial_summary)
            
        except Exception as e:
            logger.error(f"Error al generar resumen parcial: {str(e)}")
            partial_summaries.append(f"[Error en resumen de la parte {i+1}]")
    
    # Combinar resúmenes parciales en un resumen global
    combined_text = "\n\n".join(partial_summaries)
    
    try:
        # Generar resumen final
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un asistente experto en combinar resúmenes parciales en un resumen global coherente."},
                {"role": "user", "content": f"Combina estos resúmenes parciales en un único resumen global coherente:\n\n{combined_text}"}
            ],
            temperature=0.3,
            max_tokens=max_tokens
        )
        
        final_summary = completion.choices[0].message.content
        logger.info("Resumen combinado generado correctamente")
        return final_summary
        
    except Exception as e:
        logger.error(f"Error al generar resumen final combinado: {str(e)}")
        return "Error al generar el resumen combinado. " + "\n\n".join(partial_summaries[:3]) + "\n\n[...]"
