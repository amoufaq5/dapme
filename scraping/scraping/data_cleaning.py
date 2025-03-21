# scraping/data_cleaning.py

import re

def clean_text(text: str) -> str:
    """Remove unwanted characters, normalize whitespace, etc."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def process_scraped_data(scraped_data):
    """Applies cleaning to each record in scraped_data list."""
    for record in scraped_data:
        record['title'] = clean_text(record['title'])
        record['abstract'] = clean_text(record['abstract'])
    return scraped_data
