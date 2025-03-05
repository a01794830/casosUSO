import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

def summarize_global_docs(docs, model="gpt-3.5-turbo"):
    """
    Concatena varios trozos y hace un sumario global
    - si hay muchos docs, quedarte con doc[:N].
    - Se podrían hacer sumarios parciales y combinarlos.
    """
    logger.info(f"summarize_global_docs con {len(docs)} docs.")
    if not docs:
        return "No hay documentos indexados para resumir."

    text_to_summarize = "\n\n".join(docs[:20])  # top 20
    prompt = f"""Genera un sumario global de estos textos:\n\n{text_to_summarize}\n
Enfatiza los puntos clave y sé conciso.
Si no hay info, di: 'No encuentro información.'
"""
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"Eres un asistente que hace sumarios de documentos."},
                {"role":"user","content":prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        ans = completion.choices[0].message.content
        logger.debug("sumarize_global_docs completado")
        return ans
    except Exception as e:
        logger.error(f"Error en summarize_global_docs: {e}")
        return "Error generando sumario global."
