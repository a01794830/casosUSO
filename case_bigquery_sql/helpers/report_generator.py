""" Report Generator"""
import fitz  # PyMuPDF
from jinja2 import Template
import streamlit as st
import os
import pandas as pd
import math

def split_dataframe(df, rows_per_page=25):
    """Splits a DataFrame into chunks based on rows_per_page."""
    num_chunks = math.ceil(len(df) / rows_per_page)
    return [df[i * rows_per_page:(i + 1) * rows_per_page] for i in range(num_chunks)]

def generate_pdf_from_df(dataFrame, template_path="templates/table_template.html", output_filename="report.pdf", template_type="table", map_data=None):
    """Generates a PDF from HTML using PyMuPDF and handles multi-page tables or map data.
    
    Args:
        dataFrame: DataFrame to render in table template
        template_path: Path to the HTML template
        output_filename: Output PDF filename
        template_type: Type of template ('table' or 'map')
        map_data: Dictionary containing map data needed for map_template.html
    """
    try:
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        # Load HTML template
        with open(template_path, "r", encoding="utf-8") as file:
            html_template = file.read()

        # Create a new PDF document
        doc = fitz.open()

        if template_type == "table":
            # Split data into multiple pages (adjust rows_per_page as needed)
            rows_per_page = 25  # Adjust based on font size and PDF page size
            df_chunks = split_dataframe(pd.DataFrame(dataFrame), rows_per_page)

            for idx, df_chunk in enumerate(df_chunks):
                # Convert DataFrame chunk to HTML
                df_html = df_chunk.to_html(index=False, border=1)

                # Render template with data
                template = Template(html_template)
                rendered_html = template.render(table=df_html, page_number=idx + 1)

                # Add a new page
                page = doc.new_page()

                # Insert the HTML table into the PDF
                page.insert_htmlbox(fitz.Rect(50, 50, 550, 750), rendered_html)

        elif template_type == "map":
            if map_data is None:
                raise ValueError("Map data must be provided for map template")
                
            # Render map template with map data
            template = Template(html_template)
            rendered_html = template.render(**map_data)  # Pass map data as kwargs to template
            print(rendered_html)
            
            # Add a new page
            page = doc.new_page()
            
            # Insert the HTML map into the PDF
            page.insert_htmlbox(fitz.Rect(50, 50, 550, 750), rendered_html)
                
        else:
            raise ValueError(f"Unsupported template type: {template_type}")

        # Save the PDF
        doc.save(output_filename)
        doc.close()

        # ✅ Display success message in Streamlit
        st.success(f"Report generated: {output_filename}")
        return output_filename

    except Exception as e:
        st.error(f"Error generating report: {e}")
        return None