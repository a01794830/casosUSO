"""
Utilidades para operaciones de base de datos vectorial usando Pinecone.
Maneja inicialización, conexión y operaciones CRUD con índices vectoriales.
"""
import time
import logging
from typing import List, Dict, Any, Optional, Union
from pinecone import Pinecone, ServerlessSpec
from config import Config

logger = logging.getLogger(__name__)

def initialize_pinecone_index(name: str) -> Optional[Any]:
    """
    Inicializa y conecta a un índice de Pinecone.
    
    Args:
        name (str): Nombre del índice a crear o conectar
        
    Returns:
        Optional[Any]: Objeto del índice conectado o None si hay error
    """
    logger.info(f"Inicializando índice Pinecone: {name}")
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    
    try:
        # Verificar si el índice existe
        if name not in pc.list_indexes().names():
            logger.info(f"Creando nuevo índice: {name}")
            pc.create_index(
                name=name,
                dimension=Config.VECTOR_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=Config.PINECONE_CLOUD,
                    region=Config.PINECONE_REGION
                )
            )
            logger.info(f"✅ Índice '{name}' creado correctamente")
            
            # Esperar a que el índice se inicialice
            wait_count = 0
            while wait_count < 30:  # Timeout después de 30 segundos
                if pc.describe_index(name).status.get("ready", False):
                    break
                time.sleep(1)
                wait_count += 1
                
            if wait_count >= 30:
                logger.warning(f"Timeout esperando inicialización del índice {name}")
        else:
            logger.info(f"ℹ️ Índice '{name}' ya existe, conectando")
        
        # Conectar al índice
        index = pc.Index(name)
        logger.info(f"Conexión a índice {name} establecida")
        return index
        
    except Exception as e:
        logger.error(f"❌ Error con índice Pinecone: {e}", exc_info=True)
        return None


class PineconeVectorDB:
    """
    Clase para gestionar operaciones con la base de datos vectorial Pinecone.
    
    Maneja creación, conexión y operaciones CRUD con índices vectoriales de Pinecone.
    """
    def __init__(self, name: str):
        """
        Inicializa una instancia de PineconeVectorDB.
        
        Args:
            name (str): Nombre del índice de Pinecone a utilizar
        """
        logger.info(f"Inicializando PineconeVectorDB con índice: {name}")
        # Inicializar cliente Pinecone
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.name = name
        self.index = None
        self._create_and_connect_index()
        
    def _create_and_connect_index(self) -> None:
        """
        Método privado para crear y conectar al índice.
        Configura spec de acuerdo a valores de config.
        """
        try:
            # Verificar si el índice existe
            if self.name not in self.pc.list_indexes().names():
                logger.info(f"Creando índice: {self.name}")
                self.pc.create_index(
                    name=self.name,
                    dimension=Config.VECTOR_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=Config.PINECONE_CLOUD,
                        region=Config.PINECONE_REGION
                    )
                )
                logger.info("✅ Índice creado correctamente")
                
                # Esperar a que el índice esté listo
                start_time = time.time()
                while time.time() - start_time < 60:  # Esperar máximo 60 segundos
                    status = self.pc.describe_index(self.name).status
                    if status.get("ready", False):
                        break
                    time.sleep(2)
            else:
                logger.info(f"ℹ️ Índice '{self.name}' ya existe, continuando...")

            # Conectar al índice
            self.index = self.pc.Index(self.name)
            logger.info("✅ Conexión al índice establecida")
            
        except Exception as e:
            logger.error(f"❌ Error con índice: {e}", exc_info=True)
            
    def upsert_vectors(self, vectors: List[tuple]) -> bool:
        """
        Inserta o actualiza vectores en el índice.
        
        Args:
            vectors (List[tuple]): Lista de tuplas (id, vector, metadata)
            
        Returns:
            bool: Estado de éxito
        """
        if not self.index:
            logger.error("❌ No hay conexión activa al índice")
            return False
            
        if not vectors:
            logger.warning("Lista de vectores vacía, no se realizó upsert")
            return True
            
        try:
            # Procesar en lotes para evitar límites de API
            BATCH_SIZE = 100
            for i in range(0, len(vectors), BATCH_SIZE):
                batch = vectors[i:i+BATCH_SIZE]
                self.index.upsert(vectors=batch)
                logger.debug(f"Batch de {len(batch)} vectores insertado correctamente")
            
            logger.info(f"✅ {len(vectors)} vectores insertados correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en upsert_vectors: {e}", exc_info=True)
            return False
            
    def query(
        self, 
        vector: List[float], 
        top_k: int = 5, 
        include_metadata: bool = True,
        filter: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Consulta el índice para vectores similares.
        
        Args:
            vector (List[float]): Vector de consulta
            top_k (int): Número de resultados a retornar
            include_metadata (bool): Incluir metadata en resultados
            filter (Dict[str, Any]): Filtro de metadatos opcional
            
        Returns:
            Optional[Dict[str, Any]]: Resultados de la consulta o None si hay error
        """
        if not self.index:
            logger.error("❌ No hay conexión activa al índice")
            return None
            
        try:
            kwargs = {
                "vector": vector,
                "top_k": top_k,
                "include_values": False,
                "include_metadata": include_metadata
            }
            
            if filter:
                kwargs["filter"] = filter
                
            results = self.index.query(**kwargs)
            logger.debug(f"Query exitosa, {len(results.matches) if results.matches else 0} resultados")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en query: {e}", exc_info=True)
            return None
            
    def delete_vectors(self, ids: Union[str, List[str]]) -> bool:
        """
        Elimina vectores del índice.
        
        Args:
            ids (Union[str, List[str]]): ID único o lista de IDs a eliminar
            
        Returns:
            bool: Estado de éxito
        """
        if not self.index:
            logger.error("❌ No hay conexión activa al índice")
            return False
        
        # Normalizar ids a lista
        if isinstance(ids, str):
            ids = [ids]
            
        if not ids:
            logger.warning("Lista de IDs vacía, no se realizó delete")
            return True
            
        try:
            self.index.delete(ids=ids)
            logger.info(f"✅ {len(ids)} vectores eliminados correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en delete_vectors: {e}", exc_info=True)
            return False
