"""
Utilidades para la generación de reportes PDF para el sistema IoT.
"""
import io
import logging
from typing import List, Dict, Any, Optional, Union
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

def generar_pdf(
    data_list: List[Dict[str, Any]],
    titulo: str = "Reporte IoT Monitor"
) -> Optional[bytes]:
    """
    Genera un PDF en memoria con datos de dispositivos IoT.
    
    Args:
        data_list (List[Dict[str, Any]]): Lista de diccionarios con datos IoT
        titulo (str): Título del reporte
        
    Returns:
        Optional[bytes]: Contenido del PDF en bytes o None si hay error
    """
    if not data_list:
        logger.warning("No hay datos para generar el PDF")
        return None
        
    logger.info(f"Generando PDF con {len(data_list)} registros")
    
    try:
        # Crear buffer en memoria para el PDF
        buffer = io.BytesIO()
        
        # Crear canvas con tamaño de página
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Configurar metadatos del documento
        c.setTitle(titulo)
        c.setAuthor("IoT Monitor")
        c.setSubject("Reporte de dispositivos IoT")
        
        # Crear estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Title"],
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=12
        )
        
        subtitle_style = ParagraphStyle(
            name="SubtitleStyle",
            parent=styles["Heading2"],
            fontSize=12,
            alignment=TA_LEFT,
            spaceAfter=6
        )
        
        # Dibujar encabezado
        c.setFont("Helvetica-Bold", 16)
        c.drawString(0.5*inch, height - 1*inch, titulo)
        c.setFont("Helvetica", 10)
        c.drawString(0.5*inch, height - 1.2*inch, 
                    "Este reporte contiene datos de tus dispositivos IoT")
        
        # Añadir fecha y hora
        from datetime import datetime
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.drawRightString(width - 0.5*inch, height - 1*inch, f"Fecha: {fecha_actual}")
        
        # Posición inicial para contenido
        y = height - 1.5*inch
        
        # Funciones para manejar colores según valores
        def get_battery_color(level):
            try:
                level = float(level)
                if level <= 10:
                    return colors.red
                elif level <= 20:
                    return colors.orange
                elif level >= 80:
                    return colors.green
                return colors.black
            except:
                return colors.black
                
        def get_signal_color(signal):
            try:
                signal = float(signal)
                if signal <= 30:
                    return colors.red
                elif signal <= 50:
                    return colors.orange
                elif signal >= 70:
                    return colors.green
                return colors.black
            except:
                return colors.black
        
        # Procesar cada registro
        for idx, metadata in enumerate(data_list, start=1):
            # Verificar si queda espacio en la página
            if y < 2*inch:  # Si queda poco espacio
                c.showPage()  # Nueva página
                # Repetir encabezado
                c.setFont("Helvetica-Bold", 16)
                c.drawString(0.5*inch, height - 1*inch, f"{titulo} (Continuación)")
                c.setFont("Helvetica", 10)
                c.drawRightString(width - 0.5*inch, height - 1*inch, f"Página {c.getPageNumber()}")
                y = height - 1.5*inch  # Resetear posición
            
            # Encabezado de registro
            c.setFillColor(colors.darkblue)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(0.5*inch, y, f"Registro #{idx}")
            y -= 0.25*inch
            
            # Agrupar campos por categorías
            identificacion = {}
            ubicacion = {}
            estado = {}
            otros = {}
            
            for k, v in metadata.items():
                if k in ["device_id", "user_id"]:
                    identificacion[k] = v
                elif k in ["latitude", "longitude"]:
                    ubicacion[k] = v
                elif k in ["battery_level", "signal_strength", "status", "tamper_detected"]:
                    estado[k] = v
                else:
                    otros[k] = v
            
            # Dibujar datos agrupados
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(colors.black)
            c.drawString(0.75*inch, y, "Identificación:")
            
            y -= 0.2*inch
            
            c.setFont("Helvetica", 10)
            for k, v in identificacion.items():
                line = f"{k}: {v}"
                c.drawString(0.9*inch, y, line)
                y -= 0.2*inch
            
          
            # Espacio
            y -= 0.1*inch
            
            c.setFont("Helvetica-Bold", 10)
            c.drawString(0.75*inch, y, "Estado:")
            y -= 0.2*inch
            
            c.setFont("Helvetica", 10)
            for k, v in estado.items():
                # Aplicar colores según tipo de campo
                if k == "battery_level":
                    c.setFillColor(get_battery_color(v))
                    line = f"{k}: {v}%"
                elif k == "signal_strength":
                    c.setFillColor(get_signal_color(v))
                    line = f"{k}: {v}"
                elif k == "tamper_detected" and str(v).lower() in ["true", "1"]:
                    c.setFillColor(colors.red)
                    line = f"{k}: {v} ⚠️"
                else:
                    c.setFillColor(colors.black)
                    line = f"{k}: {v}"
                
                c.drawString(0.9*inch, y, line)
                c.setFillColor(colors.black)  # Restaurar color
                y -= 0.2*inch
            
            # Otros campos
            if otros:
                # Espacio
                y -= 0.1*inch
                
                c.setFont("Helvetica-Bold", 10)
                c.drawString(0.75*inch, y, "Otros datos:")
                y -= 0.2*inch
                
                c.setFont("Helvetica", 10)
                for k, v in otros.items():
                    line = f"{k}: {v}"
                    c.drawString(0.9*inch, y, line)
                    y -= 0.2*inch
            
            # Línea separadora
            c.setStrokeColor(colors.lightgrey)
            c.line(0.5*inch, y - 0.1*inch, 7.5*inch, y - 0.1*inch)
            y -= 0.3*inch
            
            # Verificar espacio para el siguiente registro
            if y < 1.5*inch and idx < len(data_list):
                c.showPage()
                # Repetir encabezado en nueva página
                c.setFont("Helvetica-Bold", 16)
                c.drawString(0.5*inch, height - 1*inch, f"{titulo} (Continuación)")
                c.setFont("Helvetica", 10)
                c.drawRightString(width - 0.5*inch, height - 1*inch, f"Página {c.getPageNumber()}")
                y = height - 1.5*inch
        
        # Pie de página en la última página
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(0.5*inch, 0.5*inch, "Generado por IoT Monitor - Sistema de monitoreo de dispositivos")
        c.drawRightString(width - 0.5*inch, 0.5*inch, f"Página {c.getPageNumber()} de {c.getPageNumber()}")
        
        # Finalizar documento
        c.save()
        
        # Obtener los bytes del PDF
        pdf_data = buffer.getvalue()
        buffer.close()
        
        logger.info(f"PDF generado correctamente: {len(pdf_data)} bytes")
        return pdf_data
        
    except Exception as e:
        logger.error(f"Error generando PDF: {e}", exc_info=True)
        return None
