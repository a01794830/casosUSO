import os
import time
import logging
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from case_documents.utils.embedding_utils import get_embedding_new

logger = logging.getLogger(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY","")
PINECONE_REGION = os.getenv("PINECONE_REGION","us-east-1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD","aws")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME","doc-rag")
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
    else:
        logger.info(f"Usando índice '{INDEX_NAME}'.")
    return pc.Index(INDEX_NAME)

def upsert_docs(chunks):
    """
    Recibe lista de trozos (strings), genera embeddings y sube a Pinecone
    """
    index = get_or_create_index()
    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        vectors = []
        for txt in batch:
            emb = get_embedding_new(txt)
            doc_id = str(uuid.uuid4())
            meta = {"TEXT": txt}
            vectors.append((doc_id, emb, meta))
        index.upsert(vectors=vectors)
        logger.debug(f"Upsert {len(batch)} trozos.")

def search_docs(query, top_k=100):
    """
    Búsqueda embeddings en Pinecone
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
        if not res or not res.matches:
            logger.info("No matches en la búsqueda.")
            return []
        docs = []
        for m in res.matches:
            txt = m.metadata.get("TEXT","")
            if txt:
                docs.append(txt)
        logger.debug(f"search_docs => {len(docs)} docs.")
        return docs
    except Exception as e:
        logger.error(f"Error en search_docs: {e}")
        return []

def get_all_docs(top_k=5000):
    """
    Descarga todos los docs (en un vector dummy).
    """
    index = get_or_create_index()
    dummy_vec = [0.0]*VECTOR_DIM
    try:
        res = index.query(
            vector=dummy_vec,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        if not res or not res.matches:
            logger.info("No matches en get_all_docs.")
            return []
        docs = []
        for m in res.matches:
            txt = m.metadata.get("TEXT","")
            if txt:
                docs.append(txt)
        logger.info(f"get_all_docs => {len(docs)} trozos totales.")
        return docs
    except Exception as e:
        logger.error(f"Error get_all_docs: {e}")
        return []
