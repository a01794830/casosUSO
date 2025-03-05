import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors

def generar_pdf(data_list):
    """
    Genera PDF en memoria con data_list (lista de dicts).
    Incluimos un encabezado y dibujamos una 'tabla' simple.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Header
    c.setTitle("Reporte IoT Monitor")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.5*inch, height - 1*inch, "Reporte IoT Monitor")
    c.setFont("Helvetica", 10)
    c.drawString(0.5*inch, height - 1.2*inch, "Este reporte contiene los datos consultados de tus dispositivos IoT")

    # Avanzamos un poco
    y = height - 1.5*inch

    for idx, metadata in enumerate(data_list, start=1):
        # Encabezado de registro
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(0.5*inch, y, f"Registro #{idx}")
        y -= 0.25*inch

        c.setFont("Helvetica", 10)
        for k, v in metadata.items():
            line = f"{k}: {v}"
            c.drawString(0.75*inch, y, line)
            y -= 0.2*inch
            if y < 1*inch:
                c.showPage()
                # Repetir header
                c.setFont("Helvetica-Bold", 16)
                c.drawString(0.5*inch, height - 1*inch, "Reporte IoT Monitor (Continuación)")
                y = height - 1.5*inch
                c.setFont("Helvetica", 10)

        c.setStrokeColor(colors.gray)
        c.line(0.5*inch, y, 7.8*inch, y)
        y -= 0.3*inch

        if y < 1*inch:
            c.showPage()
            c.setFont("Helvetica-Bold", 16)
            c.drawString(0.5*inch, height - 1*inch, "Reporte IoT Monitor (Continuación)")
            y = height - 1.5*inch
            c.setFont("Helvetica", 10)

    c.save()
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
