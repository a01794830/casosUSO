# Aplicación Demo de Casos de Uso de GenAI

Esta aplicación Streamlit demuestra cuatro casos de uso diferentes para aplicaciones de Inteligencia Artificial Generativa en contextos empresariales. Cada caso de estudio presenta una aplicación diferente de GenAI para resolver problemas prácticos.

## Casos de Uso

### Caso 1: Generación de SQL con GenAI
Demuestra el uso de GenAI para generar consultas SQL a partir de peticiones en lenguaje natural, ayudando a usuarios de negocios a acceder a datos sin conocimientos profundos de SQL.

### Caso 2: Análisis de Datos IoT
Muestra cómo se puede aplicar GenAI para analizar y visualizar datos de dispositivos IoT.

### Caso 3: Generación de Contenido Educativo
Demuestra la generación automática de materiales educativos a partir de artículos web, incluyendo resúmenes y presentaciones PowerPoint.

### Caso 4: Análisis de Documentos
Presenta técnicas RAG (Generación Aumentada por Recuperación) para analizar documentos, generar resúmenes y responder preguntas basadas en el contenido del documento.

## Primeros Pasos

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```
2. Configurar variables de entorno en el archivo .env

- OPENAI_API_KEY
- PINECONE_API_KEY (si usa Pinecone)
- BIGQUERY_PROJECT_ID (si usa BigQuery)

3. Ejecutar la aplicación:

## Tecnologías Utilizadas
- Streamlit para la interfaz web
- OpenAI para capacidades de IA generativa
- Pinecone para almacenamiento vectorial y búsqueda de similitud
- Google BigQuery para almacenamiento de datos y ejecución de SQL
- Langchain para implementación de RAG
