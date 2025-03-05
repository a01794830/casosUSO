import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

def parse_pdf(file):
    """
    Lee un archivo PDF (tipo Streamlit) y retorna el texto concatenado.
    """
    logger.info(f"parse_pdf para {file.name}")
    pdf_data = file.read()
    text = ""
    try:
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logger.error(f"Error parse_pdf: {e}")
    return text

def chunk_text(text, chunk_size=800):
    """
    Trocea en trozos de ~chunk_size caracteres.
    """
    logger.info(f"chunk_text: {len(text)} caracteres, chunk_size={chunk_size}")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    logger.info(f"chunk_text generó {len(chunks)} trozos.")
    return chunks
