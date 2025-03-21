# scraping/multi_scraper.py

import requests
from bs4 import BeautifulSoup
import time
import random
import logging

logging.basicConfig(level=logging.INFO)

class MultiSourceScraper:
    def __init__(self, sources, max_pages=3):
        """
        sources: dict mapping source name to base URL.
        max_pages: number of pages to scrape for each source.
        """
        self.sources = sources
        self.max_pages = max_pages
        self.data = {}

    def scrape_source(self, source_name, base_url):
        logging.info(f"Scraping source: {source_name}")
        source_data = []
        for page in range(1, self.max_pages + 1):
            # Append page parameter if needed. Adjust based on source URL parameters.
            full_url = f"{base_url}&page={page}"
            try:
                response = requests.get(full_url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    # Adjust the selectors according to the HTML structure of the source.
                    articles = soup.find_all("div", class_="docsum-content")
                    for article in articles:
                        title_elem = article.find("a", class_="docsum-title")
                        abstract_elem = article.find("div", class_="full-view-snippet")
                        title = title_elem.text.strip() if title_elem else "No Title"
                        abstract = abstract_elem.text.strip() if abstract_elem else "No Abstract"
                        source_data.append({
                            "source": source_name,
                            "title": title,
                            "abstract": abstract,
                            "url": full_url
                        })
                    logging.info(f"Source {source_name}: Page {page} scraped, found {len(articles)} articles.")
                else:
                    logging.warning(f"Source {source_name}: Failed to fetch {full_url} (Status: {response.status_code}).")
            except Exception as e:
                logging.error(f"Source {source_name}: Error scraping {full_url}: {e}")
            time.sleep(random.uniform(1, 3))  # Sleep to reduce load and avoid IP bans.
        return source_data

    def scrape_all(self):
        for source_name, base_url in self.sources.items():
            self.data[source_name] = self.scrape_source(source_name, base_url)
        return self.data

if __name__ == "__main__":
    # Define your sources with actual, ready-to-scrape URLs.
    sources = {
        "pubmed": "https://example.com/pubmed?term=medicine",          # Replace with actual PubMed URL
        "clinicaltrials": "https://example.com/clinicaltrials?term=medicine"  # Replace with actual ClinicalTrials URL
    }
    scraper = MultiSourceScraper(sources, max_pages=3)
    all_data = scraper.scrape_all()
    for src, articles in all_data.items():
        logging.info(f"Total articles scraped from {src}: {len(articles)}")
