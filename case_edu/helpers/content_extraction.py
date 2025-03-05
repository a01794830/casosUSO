import requests
from bs4 import BeautifulSoup

def extract_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_text = ''
    for paragraph in soup.find_all('p'):
        article_text += paragraph.text
    return article_text