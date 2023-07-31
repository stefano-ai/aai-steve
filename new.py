import os
import requests
from bs4 import BeautifulSoup
import pdfkit

def scrape_and_create_pdf(sitemap_url, output_dir='.'):
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
    except requests.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return
    except Exception as err:
        print(f'Other error occurred: {err}')
        return
    
    soup = BeautifulSoup(response.content, 'lxml')

    # Find all URLs in the sitemap
    urls = [element.text for element in soup.find_all('loc')]

    for url in urls:
        try:
            # Get the HTML content of the page
            response = requests.get(url)
            response.raise_for_status()

            # Create a PDF from the HTML content
            pdfkit.from_string(response.text, os.path.join(output_dir, f'{url.replace("/", "_")}.pdf'))
        except requests.HTTPError as http_err:
            print(f'HTTP error occurred while accessing {url}: {http_err}')
        except Exception as err:
            print(f'Other error occurred while accessing {url}: {err}')

if __name__ == "__main__":
    sitemap_url = 'https://nuvitababy.com/sitemap_products_1.xml?from=7151132639418&to=8434915180870'  # Replace with your sitemap URL
    scrape_and_create_pdf(sitemap_url)
