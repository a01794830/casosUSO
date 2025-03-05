import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

def get_embedding_new(text:str, model="text-embedding-ada-002"):
    """
    Genera embedding con openai>=1.0.0
    """
    logger.debug(f"get_embedding_new con model={model}, len(text)={len(text)}")
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.embeddings.create(
            input=text,
            model=model
        )
        emb = resp.data[0].embedding
        return emb
    except Exception as e:
        logger.error(f"Error get_embedding_new: {e}")
        # devolvemos vector de 1536 zeros
        return [0.0]*1536

def generate_chat_response(contexts, query, model="gpt-3.5-turbo"):
    """
    contexts: list[str]
    query: str
    Toma top 3 contexts
    """
    logger.debug(f"generate_chat_response con {len(contexts)} contextos, query={query[:50]}...")
    joined_context = "\n\n---\n\n".join(contexts[:3])  # top 3
    prompt = f"""Basado en este texto:\n{joined_context}\n
Pregunta: {query}
Si no hay info, di: "No encuentro esa información."
    """
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"Eres un asistente sobre documentos subidos."},
                {"role":"user","content":prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        ans = completion.choices[0].message.content
        logger.info("generate_chat_response ok")
        return ans
    except Exception as e:
        logger.error(f"Error generate_chat_response: {e}")
        return "Error generando respuesta."
