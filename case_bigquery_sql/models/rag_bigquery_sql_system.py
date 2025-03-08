"""
Sistema RAG para generación de consultas SQL desde lenguaje natural.
Utiliza embeddings, Pinecone, OpenAI y BigQuery.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from google.cloud import bigquery
from case_bigquery_sql.helpers.embeddings_helper import get_openai_embeddings
from config import Config

logger = logging.getLogger(__name__)

# Constantes para la configuración BigQuery
DATASET_ID = Config.DATASET_ID
TABLE_ID = Config.TABLE_ID
FIELDS = "device_id,user_id,latitude,longitude,battery_level,signal_strength,tamper_detected,status,restriction_violation,timestamp"

class RAGBigQuerySystem:
    """
    Sistema de Generación Aumentada por Recuperación (RAG) para consultas SQL en BigQuery.
    
    Esta clase proporciona funcionalidades para convertir consultas en lenguaje natural
    a consultas SQL optimizadas para BigQuery, utilizando técnicas de RAG con Pinecone
    y modelos de lenguaje de OpenAI.
    """
    
    def __init__(self, index):
        """
        Inicializa el sistema RAG.
        
        Args:
            index: Índice Pinecone para almacenar y recuperar ejemplos
        """
        self.index = index
        self.llm = OpenAI(temperature=0.3, openai_api_key=Config.OPENAI_API_KEY)
        logger.info("RAGBigQuerySystem inicializado")

    def _store_query(self, query: str, metadata: Dict[str, str]) -> None:
        """
        Almacena una consulta y su embedding en Pinecone para recuperación futura.
        
        Args:
            query (str): Consulta en lenguaje natural
            metadata (Dict[str, str]): Metadatos como la consulta SQL generada
        """
        if not query or not metadata:
            logger.warning("Intento de almacenar consulta vacía o sin metadatos")
            return
            
        try:
            # Generar embedding para la consulta
            vector = get_openai_embeddings(query)[0]
            
            # Preparar vector para Pinecone (id, vector, metadata)
            vector_record = (query, vector, metadata)
            
            # Insertar en Pinecone
            self.index.upsert(vectors=[vector_record])
            logger.info(f"Consulta almacenada: '{query[:30]}...'")
            
        except Exception as e:
            logger.error(f"Error al almacenar consulta: {e}")

    def _search_similar_query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Recupera consultas similares previamente almacenadas desde Pinecone.
        
        Args:
            query (str): Consulta en lenguaje natural para buscar similares
            top_k (int): Número máximo de resultados a retornar
            
        Returns:
            List[Dict[str, Any]]: Lista de matches con metadatos incluidos
        """
        try:
            # Generar embedding para la consulta
            query_embedding = get_openai_embeddings(query)[0]
            
            # Verificar si el índice está inicializado
            if self.index is None:
                logger.warning("Índice Pinecone no inicializado")
                return []
                
            # Realizar búsqueda en Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extraer matches
            if not results or not hasattr(results, 'matches'):
                logger.info("No se encontraron consultas similares")
                return []
                
            logger.info(f"Se encontraron {len(results.matches)} consultas similares")
            return results.matches
            
        except Exception as e:
            logger.error(f"Error al buscar consultas similares: {e}")
            return []
    
    def generate_sql_query(self, user_input: str) -> str:
        """
        Genera una consulta SQL a partir de una entrada en lenguaje natural.
        
        Utiliza ejemplos previos similares y un modelo de lenguaje para crear
        la consulta SQL optimizada para BigQuery.
        
        Args:
            user_input (str): Consulta en lenguaje natural
            
        Returns:
            str: Consulta SQL generada
        """
        logger.info(f"Generando SQL para: '{user_input}'")
        
        # Recuperar consultas similares como ejemplos
        past_queries = self._search_similar_query(user_input, top_k=3)
        
        # Construir contexto con ejemplos similares
        context = ""
        if past_queries:
            context += "Ejemplos de consultas similares:\n\n"
            for match in past_queries:
                if 'metadata' in match and 'query' in match.metadata and 'sql' in match.metadata:
                    query_text = match.metadata['query']
                    sql = match.metadata['sql']
                    context += f"Consulta: {query_text}\nSQL: {sql}\n\n"
        
        logger.debug(f"Contexto construido con {len(past_queries)} ejemplos")
        
        # Plantilla de prompt para generación SQL
        prompt = PromptTemplate(
            input_variables=["context", "user_input", "DATASET_ID", "TABLE_ID"],
            template="""
            # Genera una consulta SQL para BigQuery que responda a la siguiente pregunta: 
            # {user_input}
            
            # Ejemplos relevantes de SQL y patrones que pueden ayudar:
            {context}
            
            # La consulta debe seguir estos requisitos:
            # Seguridad:
            # 1. Solo usar operaciones SELECT (no INSERT, UPDATE, DELETE)
            # 2. Solo acceder a la tabla tracking_dataset.tracking_data
            # 3. Usar solo estos campos disponibles:
               - device_id (string): identificador único del dispositivo
               - user_id (string): identificador del usuario asociado al dispositivo
               - latitude (float): coordenada geográfica de latitud
               - longitude (float): coordenada geográfica de longitud
               - battery_level (integer): porcentaje de batería restante (0-100)
               - signal_strength (integer): fuerza de señal celular (0-100)
               - tamper_detected (boolean): True si se detectó manipulación del dispositivo
               - status (integer): estado actual del dispositivo ('active' = 1, 'inactive'= 2, 'error' = 3)
               - restriction_violation (boolean): True si el dispositivo está fuera de la geozona permitida
               - timestamp (datetime): tiempo cuando se registraron los datos

            Reglas importantes:
            - Incluir siempre una cláusula WHERE para limitar resultados
            - Limitar resultados a 1000 filas por defecto a menos que se especifique lo contrario
            - Usar alias claros para columnas calculadas
            - Formatear la consulta SQL con indentación adecuada y saltos de línea
            - Devolver nombres de campos en formato legible, p.ej. battery_level as "Nivel de Batería"

            Escenarios comunes y su implementación:
            1. Batería baja:
               - "Batería baja" significa battery_level <= 20
               - "Batería crítica" significa battery_level <= 10
               - "Batería buena" significa battery_level >= 80

            2. Categorías de fuerza de señal:
               - "Señal pobre" significa signal_strength <= 30
               - "Buena señal" significa signal_strength >= 70

            3. Consultas basadas en tiempo:
               - "Reciente" significa dentro de las últimas 24 horas

            # Basado en la solicitud del usuario y los ejemplos anteriores, genera la consulta SQL más apropiada:
            """
        )
        
        # Generar la consulta SQL utilizando el modelo de lenguaje
        try:
            sql_query = self.llm(prompt.format(
                user_input=user_input,
                context=context,
                DATASET_ID=DATASET_ID,
                TABLE_ID=TABLE_ID
            ))
            
            # Almacenar la consulta y el SQL generado para futuras recuperaciones
            metadata = {
                "query": user_input,
                "sql": sql_query
            }
            self._store_query(user_input, metadata)
            
            logger.info("SQL generado correctamente")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error al generar SQL: {e}")
            return f"Error al generar consulta SQL: {str(e)}"

    def get_schema_for_prompt(self) -> str:
        """
        Devuelve una representación formateada del esquema de tabla para incluir en prompts LLM.
        
        Returns:
            str: Información de esquema formateada como string
        """
        schema_info = self.get_table_details()
        if not schema_info:
            return "Información de esquema no disponible."
        
        schema_text = f"Tabla: {schema_info['dataset']}.{schema_info['table_name']}\n\n"
        schema_text += "Campos disponibles:\n"
        
        for field in schema_info["fields"]:
            description = field["description"] if field["description"] else "Sin descripción"
            schema_text += f"- {field['name']} ({field['type']}): {description}\n"
        
        return schema_text

    def get_table_details(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene el esquema y detalles de la tabla desde BigQuery.
        
        Returns:
            Optional[Dict[str, Any]]: Diccionario con información del esquema o None si hay error
        """
        bg_client = bigquery.Client(project=Config.BIGQUERY_PROJECT_ID)
        table_id = f"{Config.BIGQUERY_PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
        
        try:
            # Obtener referencia de tabla
            table = bg_client.get_table(table_id)
            
            # Extraer información de esquema
            schema_info = {
                "table_name": TABLE_ID,
                "dataset": DATASET_ID,
                "num_rows": table.num_rows,
                "fields": []
            }
            
            # Extraer información de campos
            for field in table.schema:
                field_info = {
                    "name": field.name,
                    "type": field.field_type,
                    "description": field.description,
                    "mode": field.mode  # 'NULLABLE', 'REQUIRED', o 'REPEATED'
                }
                schema_info["fields"].append(field_info)
                
            logger.info(f"Detalles de tabla obtenidos: {len(schema_info['fields'])} campos encontrados")
            return schema_info
            
        except Exception as e:
            logger.error(f"Error al obtener detalles de tabla: {e}")
            return None

    def query_bigquery(self, query: str) -> Optional[Any]:
        """
        Ejecuta una consulta en BigQuery y devuelve los resultados.
        
        Args:
            query (str): Consulta SQL a ejecutar
            
        Returns:
            Optional[pandas.DataFrame]: DataFrame con resultados o None si hay error
        """
        logger.info(f"Ejecutando consulta BigQuery: {query[:100]}...")
        bg_client = bigquery.Client(project=Config.BIGQUERY_PROJECT_ID)
        
        try:
            # Sanitizar query (solo permitir SELECT)
            if not query.strip().upper().startswith("SELECT"):
                raise ValueError("Solo se permiten consultas SELECT")
                
            # Ejecutar query
            result = bg_client.query(query).to_dataframe()
            logger.info(f"Consulta ejecutada exitosamente: {len(result)} filas devueltas")
            return result
            
        except Exception as e:
            logger.error(f"Error de BigQuery: {e}")
            return None
