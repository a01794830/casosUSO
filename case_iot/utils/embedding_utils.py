import os
import logging
from openai import OpenAI
import streamlit as st

logger = logging.getLogger(__name__)

def get_embedding_new(text: str, model="text-embedding-3-small"):
    """
    Genera embedding con openai>=1.0.0
    """
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=text,
            model=model,
            encoding_format="float"
        )
        emb = response.data[0].embedding
        return emb
    except Exception as e:
        logger.error(f"Error generando embedding: {e}")
        raise

def generate_chat_response(contexts, user_query, model="gpt-3.5-turbo"):
    joined_context = "\n\n---\n\n".join(contexts)
    prompt = f"""Usa el siguiente contexto de dispositivos IoT:
    {joined_context}
    ---
    PREGUNTA: {user_query}
    Si no ves la info, indica: "No tengo esa información."
    """

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un asistente especializado en IoT Monitor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generando respuesta de chat: {e}")
        return f"Error generando respuesta: {str(e)}"
