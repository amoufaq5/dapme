# scraping/scraper.py

import requests
from bs4 import BeautifulSoup
import time
import logging
import random

logging.basicConfig(level=logging.INFO)

class MedicalScraper:
    def __init__(self, base_url, max_pages=5):
        self.base_url = base_url
        self.max_pages = max_pages
        self.data = []

    def scrape(self):
        logging.info("Starting scrape process...")
        for page_number in range(1, self.max_pages + 1):
            full_url = f"{self.base_url}?page={page_number}"
            try:
                response = requests.get(full_url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    articles = soup.select("div.docsum-content")
                    for article in articles:
                        title_elem = article.select_one("a.docsum-title")
                        abstract_elem = article.select_one("div.full-view-snippet")
                        title = title_elem.text.strip() if title_elem else None
                        abstract = abstract_elem.text.strip() if abstract_elem else None
                        if title and abstract:
                            self.data.append({"title": title, "abstract": abstract})
                    logging.info(f"Scraped page {page_number}, found {len(articles)} articles.")
                else:
                    logging.warning(f"Failed to fetch {full_url}, status: {response.status_code}")
            except Exception as e:
                logging.error(f"Error scraping {full_url}: {e}")
            
            # Sleep to avoid being flagged as bot (ethical scraping).
            time.sleep(random.uniform(1, 3))

        return self.data

if __name__ == "__main__":
    # Example usage:
    base_url = "https://example.com/pubmed-like-endpoint"
    scraper = MedicalScraper(base_url, max_pages=3)
    scraped_data = scraper.scrape()
    logging.info(f"Total articles scraped: {len(scraped_data)}")
    
    # Next step would be to store them in the database or a CSV file
    # ...
