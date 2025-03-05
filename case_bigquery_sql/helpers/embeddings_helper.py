""" Embeddings OpenAI Helper """
import openai
from openai import OpenAI
from config import Config
client = OpenAI(api_key=Config.OPENAI_API_KEY)

def get_openai_embeddings(texts, model="text-embedding-3-small"):
    """
    Genera embeddings para una lista de textos utilizando OpenAI.
    """
    texts = [text.replace("\n", " ") for text in texts]  # Limpieza básica

    response = openai.embeddings.create(
        model=model,
        input=texts,   # Procesa la lista completa de textos
    )

    return [item.embedding for item in response.data]  # Extrae embeddings