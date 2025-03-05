from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from case_bigquery_sql.helpers.embeddings_helper import get_openai_embeddings
from config import Config
from google.cloud import bigquery
import logging

# AI_API_KEY)
# embedding_model = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
DATASET_ID = "tracking_dataset"
TABLE_ID = "tracking_data"
# query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
FIELDS="device_id,user_id,latitude,longitude,battery_level,signal_strength,tamper_detected,status,restriction_violation,timestamp"


""""
RAG Biquery SQL system

"""



class RAGBigQuerySystem:
    """
        Class
    """
    def __init__(self,index):
        self.index =index
        self.llm = OpenAI(temperature=0.3, openai_api_key=Config.OPENAI_API_KEY)

    def _store_query(self,query, metadata):
        """Stores a query embedding in Pinecone."""
        vector = get_openai_embeddings(query)
        # vector = embedding_model.embed_query(query)
        self.index.upsert(vectors=list((query,vector,metadata)))

    def _search_similar_query(self,query, top_k=3):
        """Retrieves similar past queries from Pinecone."""
        #vector = embedding_model.embed_query(query)
        vector = get_openai_embeddings(query)
        if self.index is None:
            print("Warning: Pinecone index not initialized")
            return []
        results = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True  # Incluir metadata de los resultados
        )
        return results["matches"]
    
    def generate_sql_query(self,user_input):
        """Uses LLM to convert natural language input to SQL query."""
        # # Retrieve similar past queries
        past_queries = self._search_similar_query(user_input, top_k=3)
        # best_match = ""
        # if past_queries:
        #     best_match = past_queries[0]['metadata']['query']
            # st.write(f"Found a similar report: {best_match}")
        context = ""
        if past_queries:
            print(past_queries)
            context += "Here are some similar past queries:\n\n"
            for match in past_queries:
                if 'metadata' in match and 'query' in match['metadata'] and 'sql' in match['metadata']:
                    query_text = match['metadata']['query']
                    sql = match['metadata']['sql']
                    context += f"User query: {query_text}\nSQL: {sql}\n\n"
        
        print('examples',context)
        # st.write(f"Found a similar report: {examples}")
        # # Extract examples from similar queries
        # examples = ""
        # for match in similar_queries:
        #     query = match.id
        #     sql = match.metadata.get("sql", "")
        #     examples += f"User query: {query}\nSQL: {sql}\n\n"
        #     # print(examples)
        # print(examples)
        # Here are some similar questions and their SQL queries:
        #     {examples}
        prompt = PromptTemplate(
            input_variables=["context","user_input","DATASET_ID","TABLE_ID"],
            template="""
            
        # Generate a BigQuery SQL query to answer the following question: 
        # {user_input}
        
        # Here are some relevant SQL examples and patterns that might help:
        {context}
        
        
        #  The query must follow these requirements:
        # Enforce security by:
        # 1. Only use SELECT operations (no INSERT, UPDATE, DELETE)
        # 2. Only access the tracking_dataset.tracking_data table
        # 3. Use only these available fields:
           - device_id (string): unique identifier for the tracking device
           - user_id (string): identifier of the user associated with the device
           - latitude (float): geographic latitude coordinate
           - longitude (float): geographic longitude coordinate
           - battery_level (integer): remaining battery percentage (0-100)
           - signal_strength (integer): cellular signal strength (0-100)
           - tamper_detected (boolean): True if device tampering was detected
           - status (inteintegerguer): current device status ('active' = 1, 'inactive'= 2, 'error' = 3)
           - restriction_violation (boolean): True if device is outside permitted geozone
           - timestamp (datetime): time when the data was recorded
        # Available fields:
            device_id (STRING): No description
            user_id (STRING): No description
            latitude (FLOAT): No description
            longitude (FLOAT): No description
            battery_level (INTEGER): No description
            signal_strength (INTEGER): No description
            tamper_detected (BOOLEAN): No description
            status (INTEGER): No description
            restriction_violation (BOOLEAN): No description
            timestamp (DATETIME): No description

        Important rules:
        - Always include a WHERE clause to limit results
        - Limit results to 1000 rows by default unless specified otherwise
        - Use clear aliasing for computed columns
        - Format the SQL query with proper indentation and line breaks
        - Return field names in a readable format e.g battery_level as Battery

        Common scenarios and their implementations:
        1. Low battery queries:
           - "Low battery" means battery_level <= 20
           - "Critical battery" means battery_level <= 10
           - "Good battery" means battery_level >= 80

        2. Signal strength categories:
           - "Poor signal" means signal_strength <= 30
           - "Good signal" means signal_strength >= 70

        3. Time-based queries:
           - "Recent" means within last 24 hours

        

        Example queries:
        1. "Give me all devices with low battery":
        SELECT
            device_id as Device,
            user_id as User,
            battery_level as Battery,
            status as Status,
            timestamp as Time
        FROM tracking_dataset.tracking_data
        WHERE battery_level <= 20
        ORDER BY battery_level ASC
        LIMIT 1000

        2. "Show devices with poor signal and errors":
        SELECT
            device_id as Device,
            user_id as User,
            battery_level as Battery,
            status as Status,
            timestamp as Time
        FROM tracking_dataset.tracking_data
        WHERE signal_strength <= 30
            AND status = 0
        ORDER BY signal_strength ASC
        LIMIT 1000

        # Based on the user's request and the examples above, generate the most appropriate SQL query:
            """
        )
        sql_query = self.llm(prompt.format(user_input=user_input,context=context,DATASET_ID=DATASET_ID,TABLE_ID=TABLE_ID))
        # Store the query and generated SQL in the index for future retrieval
        # metadata = {
        #     "query": user_input,
        #     "sql": sql_query
        # }
        # self._store_query(user_input, metadata)
        
        return sql_query
    def get_schema_for_prompt(self):
        """
        Returns a formatted string representation of the table schema for inclusion in LLM prompts.
        
        Returns:
            str: A string containing formatted table schema information.
        """
        schema_info = self.get_table_details()
        if not schema_info:
            return "Schema information not available."
        
        schema_text = f"Table: {schema_info['dataset']}.{schema_info['table_name']}\n\n"
        schema_text += "Available fields:\n"
        
        for field in schema_info["fields"]:
            description = field["description"] if field["description"] else "No description"
            schema_text += f"- {field['name']} ({field['type']}): {description}\n"
        
        return schema_text

    def get_table_details(self):
        """
        Retrieves the table schema and details from BigQuery.
        
        Returns:
            dict: A dictionary containing table schema information including fields, types, and descriptions.
        """
        bg_client = bigquery.Client(project=Config.BIGQUERY_PROJECT_ID)
        table_id = f"{Config.BIGQUERY_PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
        
        try:
            # Get the table reference
            table = bg_client.get_table(table_id)
            
            # Extract schema information
            schema_info = {
                "table_name": TABLE_ID,
                "dataset": DATASET_ID,
                "num_rows": table.num_rows,
                "fields": []
            }
            
            # Extract field information
            for field in table.schema:
                field_info = {
                    "name": field.name,
                    "type": field.field_type,
                    "description": field.description,
                    "mode": field.mode  # 'NULLABLE', 'REQUIRED', or 'REPEATED'
                }
                schema_info["fields"].append(field_info)
                
            return schema_info
        except Exception as e:
            logging.error(f"Error retrieving table details: {e}")
            return None

    def query_bigquery(self,query: str):
        """
            Big query 
        """
        bg_client = bigquery.Client(project=Config.BIGQUERY_PROJECT_ID)
        try:
            result = bg_client.query(query).to_dataframe()
            return result
        except Exception as e:
            logging.error(f"BigQuery Error: {e}")
            return None