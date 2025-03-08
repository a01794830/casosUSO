"""
Utilidades para procesamiento de documentos PDF y texto.
Proporciona funciones para extraer y segmentar contenido.
"""
import fitz  # PyMuPDF
import logging
import re
from typing import List, Optional, BinaryIO, Union

logger = logging.getLogger(__name__)

def parse_pdf(file: BinaryIO) -> str:
    """
    Lee un archivo PDF (tipo Streamlit) y retorna el texto concatenado.
    
    Args:
        file (BinaryIO): Objeto de archivo PDF subido a través de Streamlit
        
    Returns:
        str: Texto extraído del PDF
    """
    logger.info(f"Procesando PDF: {file.name}")
    
    try:
        # Leer los datos del archivo
        pdf_data = file.read()
        file.seek(0)  # Reiniciar el puntero del archivo
        
        # Extraer texto
        text = ""
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            logger.info(f"Documento abierto. Tiene {len(doc)} páginas.")
            
            for page_num, page in enumerate(doc):
                logger.debug(f"Procesando página {page_num+1}/{len(doc)}")
                page_text = page.get_text()
                text += page_text + "\n\n"
                
        logger.info(f"Texto extraído: {len(text)} caracteres")
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error al procesar PDF {file.name}: {str(e)}", exc_info=True)
        return f"ERROR: No se pudo procesar el documento {file.name}. Error: {str(e)}"

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Divide un texto en fragmentos de tamaño aproximado con superposición.
    
    Args:
        text (str): Texto a dividir en fragmentos
        chunk_size (int): Tamaño aproximado de cada fragmento en caracteres
        overlap (int): Número de caracteres de superposición entre fragmentos
        
    Returns:
        List[str]: Lista de fragmentos de texto
    """
    logger.info(f"Fragmentando {len(text)} caracteres con tamaño={chunk_size}, superposición={overlap}")
    
    if not text or len(text) == 0:
        logger.warning("Texto vacío proporcionado para fragmentación")
        return []
        
    chunks = []
    start = 0
    
    # Función para encontrar un buen punto de corte (final de frase/párrafo)
    def find_good_break_point(s: str, target_pos: int) -> int:
        # Buscar un punto, un salto de línea o un punto y coma cercano
        break_chars = ['. ', '.\n', '\n\n', '; ', '? ', '! ']
        
        # Buscar primero hacia adelante (hasta chunk_size + 100)
        search_area = s[target_pos:min(target_pos + 100, len(s))]
        for char in break_chars:
            pos = search_area.find(char)
            if pos != -1:
                return target_pos + pos + len(char)
                
        # Si no se encuentra, buscar hacia atrás (hasta 100 caracteres antes)
        search_area = s[max(0, target_pos - 100):target_pos]
        for char in break_chars:
            pos = search_area.rfind(char)
            if pos != -1:
                return target_pos - (len(search_area) - pos - len(char))
                
        # Si aún no hay un buen punto, cortar en el espacio más cercano
        forward_space = s.find(' ', target_pos)
        if forward_space != -1 and forward_space < target_pos + 50:
            return forward_space + 1
            
        # En último caso, simplemente cortar en la posición exacta
        return target_pos
    
    # Iterar mientras no lleguemos al final del texto
    while start < len(text):
        # Calcular el final previsto de este fragmento
        end = min(start + chunk_size, len(text))
        
        # Si no estamos al final del texto, buscar un buen punto de corte
        if end < len(text):
            end = find_good_break_point(text, end)
            
        # Extraer el fragmento y limpiarlo
        chunk = text[start:end].strip()
        
        # Solo añadir fragmentos no vacíos
        if chunk:
            # Limpiar espacios múltiples y caracteres extraños
            chunk = re.sub(r'\s+', ' ', chunk)
            chunks.append(chunk)
            
        # Avanzar al siguiente punto de inicio con superposición
        start = max(start + 1, end - overlap)
    
    logger.info(f"Se generaron {len(chunks)} fragmentos")
    return chunks
