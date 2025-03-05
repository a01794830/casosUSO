import os
import time
import logging
import json
import streamlit as st

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from case_iot.utils.embedding_utils import get_embedding_new

load_dotenv()
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY","")
PINECONE_REGION = os.getenv("PINECONE_REGION","us-east-1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD","aws")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME","tracking1-rag")
VECTOR_DIM = 1536

pc = Pinecone(api_key=PINECONE_API_KEY)

def get_or_create_index():
    spec = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    existing = pc.list_indexes().names()
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=spec
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        logger.info(f"Creado índice '{INDEX_NAME}' en Pinecone.")
    return pc.Index(INDEX_NAME)

def interpret_and_search(user_query: str, top_k=2000, re_rank_top=200):
    """
    1) Llamar a un LLM pidiendo un JSON con filtros (battery_level, tamper_detected, user_id, device_id, status, etc.)
    2) parsear el JSON. Si parse OK => filtrar local o con eq, etc.
    3) Sino => fallback a embedding RAG
    Retorna (list_of_texts, used_fallback)
    """
    filter_json_str = call_llm_to_get_filterJSON(user_query)
    logger.info(f"LLM interpretó la query con JSON: {filter_json_str}")

    # parse JSON
    try:
        filter_dict = json.loads(filter_json_str)
        # Aplica filter si es válido
        docs = apply_filter(filter_dict, top_k)
        if docs:
            # re-rank docs con la query
            final_contexts = re_rank_in_batches(docs, user_query, re_rank_top)
            return final_contexts, True, False
        else:
            # fallback
            return embedding_search(user_query, top_k, re_rank_top), False, True
    except Exception as e:
        logger.warning(f"Error parse JSON => fallback. {e}")
        # fallback
        return embedding_search(user_query, top_k, re_rank_top), False, True

def call_llm_to_get_filterJSON(query):
    """
    Usa OpenAI new SDK (>=1.0.0) para “interpretar” la query y retornar un JSON de filtros.
    p.ej. 
    {
      "device_id": {"$eq": "6cfc7a7a"},
      "battery_level": {"$lt": 20}
    }
    Si no hay nada, “{}”
    """
    system_prompt = """
Eres un parser de queries para IoT. 
Dados la pregunta de usuario, devuelves un JSON con los filtros 
que se puedan inferir. 
Ejemplo de JSON:
{
  "device_id": {"$eq": "abc123"},
  "battery_level": {"$lt": 10},
  "status": {"$eq": 1},
  "user_id": {"$eq": "xxyyzz"},
  "tamper_detected": {"$eq": true}
}

Si no se detecta nada, responde "{}".
No agregues texto extra, solo el JSON.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # o gpt-3.5-turbo, etc.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=200
        )
        raw_txt = completion.choices[0].message.content.strip()
        # Debería ser un JSON
        return raw_txt
    except Exception as e:
        logger.error(f"Error call_llm_to_get_filterJSON: {e}")
        return "{}"

def apply_filter(filter_dict, top_k=2000):
    """
    Aplica filter en Pinecone. 
    Si es $eq => se puede usar filter metadata eq. 
    Si es $lt => parse local. 
    ...
    Retorna lista TEXT
    """
    index = get_or_create_index()
    # Llamada masiva
    dummy_vec = [0.0]*VECTOR_DIM
    try:
        res = index.query(
            vector=dummy_vec,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        if not (res and res.matches):
            return []

        # Filtrado local
        matched_docs = []
        for match in res.matches:
            md = match.metadata
            # Verificamos si pasa todos los filtros
            if passes_filter(md, filter_dict):
                txt = md.get("TEXT","")
                if txt:
                    matched_docs.append(txt)
        return matched_docs
    except Exception as e:
        logger.error(f"Error apply_filter: {e}")
        return []

def passes_filter(md, filter_dict):
    """
    Verifica si md pasa todos los criterios. 
    Ej. 
      "battery_level": {"$lt": 10}
      "device_id": {"$eq": "6cfc7a7a"}
      "tamper_detected": {"$eq": true}
    """
    for field, condition in filter_dict.items():
        # condition es dict {"$eq": x} o {"$lt": x} etc.
        if "$eq" in condition:
            val = condition["$eq"]
            if not check_eq(md, field, val):
                return False
        elif "$lt" in condition:
            val = condition["$lt"]
            if not check_lt(md, field, val):
                return False
        # Podrías agregar mas elif "$gt", etc
    return True

def check_eq(md, field, val):
    """
    Si val es boolean => tamper_detected, parse la string
    Si val es numero => parse
    Si val es string => eq string
    """
    meta_val_str = md.get(field, None)
    if meta_val_str is None:
        return False
    # handle boolean
    if isinstance(val, bool):
        # meta_val_str => "True" o "False"
        return (meta_val_str.lower() == str(val).lower())
    # handle numeric
    if isinstance(val, (int,float)):
        # parse meta_val_str
        try:
            num = float(meta_val_str)
            return (num == val)
        except:
            return False
    # handle string
    return (meta_val_str == val)

def check_lt(md, field, val):
    """
    Val => numeric. 
    parse metadata => compare
    """
    meta_val_str = md.get(field, None)
    if meta_val_str is None:
        return False
    try:
        num = float(meta_val_str)
        return (num < val)
    except:
        return False

def embedding_search(query, top_k=2000, re_rank_top=200):
    """
    Fallback: embeddings + re-rank
    """
    index = get_or_create_index()
    try:
        q_emb = get_embedding_new(query)
        res = index.query(
            vector=q_emb,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        if not (res and res.matches):
            return []
        docs = []
        for m in res.matches:
            txt = m.metadata.get("TEXT","")
            if txt:
                docs.append(txt)
        return re_rank_in_batches(docs, query, re_rank_top)
    except Exception as e:
        logger.error(f"Error embedding_search: {e}")
        return []

def re_rank_in_batches(docs, query, top_n=200):
    """
    Evitar "Document length '1000' exceeded documents limit of 100" 
    => partimos docs en batches de 100 -> partial re-rank -> union -> re-rank final
    """
    chunk_size = 100
    partial_top = 30
    partial_res = []

    for i in range(0, len(docs), chunk_size):
        chunk = docs[i:i+chunk_size]
        chunk_reranked = re_rank_once(chunk, query, partial_top)
        partial_res.extend(chunk_reranked)

    if len(partial_res) <= 100:
        return re_rank_once(partial_res, query, top_n)
    else:
        return re_rank_in_batches(partial_res, query, top_n)

def re_rank_once(docs, query, top_n):
    if not docs:
        return []
    top_n = min(top_n, len(docs))
    try:
        rr = pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=docs,
            top_n=top_n,
            return_documents=True
        )
        if not rr or not rr.data:
            return []
        out = []
        for d in rr.data:
            out.append(d["document"]["text"])
        return out
    except Exception as e:
        logger.error(f"Error re_rank_once: {e}")
        return []

def query_by_id(input_id: str, tipo="device_id"):
    """
    Filtra en Pinecone por device_id o user_id.
    """
    index = get_or_create_index()
    dummy_vec = [0.0]*VECTOR_DIM
    my_filter = {tipo: {"$eq": input_id}}
    try:
        # top_k grande para no perder resultados
        res = index.query(
            vector=dummy_vec,
            top_k=5000,
            include_values=False,
            include_metadata=True,
            filter=my_filter
        )
        if res and res.matches:
            return [m.metadata for m in res.matches]
        return []
    except Exception as e:
        logger.error(f"Error en query_by_id: {e}")
        raise