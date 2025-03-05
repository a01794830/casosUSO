"""
    Case 3 - Educational 
"""
import streamlit as st
from case_edu.helpers.content_extraction import extract_article_content
from case_edu.helpers.text_summarization import summarize_text
from case_edu.helpers.content_generation import generate_content
from case_edu.helpers.slide_creation import create_presentation
from case_edu.helpers.video_creation import create_video
import os 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from config import Config
st.title("Caso 3 - Generacion de Material didactico atravez de Articulos")

url = st.text_input("Enter the URL of the article:")

if st.button("Create Presentation and Video"):
    with st.spinner('Extracting article content...'):
        article_content = extract_article_content(url)
        st.write(article_content)
    with st.spinner('Summarizing text...'):
        summary = summarize_text(article_content)
        st.write(summary)
    with st.spinner('Generating content...'):
        slides_content = generate_content(summary)
        st.write(slides_content)
    with st.spinner('Creating presentation...'):
        create_presentation(slides_content)
    # with st.spinner('Creating video...'):
    #     create_video(slides_content)
    
    st.success('Presentation and video created successfully!')
    with open('presentation.pptx', 'rb') as file:
        pptx_data = file.read()
        # Display download button with actual file data
        st.download_button(
            label='Download Presentation',
            data=pptx_data,
            file_name='results/presentation.pptx',
            mime='application/vnd.openxmlformats-officedocument.presentationml.presentation'
        )
        st.info("📊 PPTX files cannot be directly previewed in Streamlit. Please download the file to view the presentation.")

    # st.download_button('Download Presentation', 'presentation.pptx')
    # st.download_button('Download Video', 'presentation.avi')