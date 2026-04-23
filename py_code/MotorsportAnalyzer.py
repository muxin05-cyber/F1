from bs4 import BeautifulSoup
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from ArticleAnalyzer import ArticleAnalyzer


class MotorsportAnalyzer(ArticleAnalyzer):
    def __init__(self, period=5):
        self.url = "https://www.motorsport.com/sitemaps/news.xml"
        self.period = period
        super().__init__(self.url, self.period)

    def get_links(self):
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка загрузки sitemap: {e}")
            return []

        root = ET.fromstring(response.content)
        namespaces = {
            'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9',
            'news': 'http://www.google.com/schemas/sitemap-news/0.9'
        }

        cutoff = datetime.now() - timedelta(days=self.period)
        articles = []

        for url_elem in root.findall('.//ns:url', namespaces=namespaces):
            loc = url_elem.find('ns:loc', namespaces=namespaces)
            if loc is None or loc.text is None:
                continue
            if '/f1/' not in loc.text.lower():
                continue

            news_elem = url_elem.find('news:news', namespaces=namespaces)
            if news_elem is None:
                continue

            pub_date_elem = news_elem.find('news:publication_date', namespaces=namespaces)
            title_elem = news_elem.find('news:title', namespaces=namespaces)

            if pub_date_elem is not None and pub_date_elem.text:
                try:
                    pub_date = datetime.fromisoformat(
                        pub_date_elem.text.replace('Z', '+00:00')
                    ).replace(tzinfo=None)

                    if pub_date >= cutoff:
                        articles.append({
                            'url': loc.text,
                            'title': title_elem.text if title_elem is not None else "",
                            'date': pub_date.strftime('%Y-%m-%d')
                        })
                except (ValueError, AttributeError):
                    continue

        return articles

    def get_text_by_link(self, single_link):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            response = requests.get(single_link, headers=headers, timeout=30)
            response.raise_for_status()
        except:
            return ""

        soup = BeautifulSoup(response.text, 'lxml')

        possible_selectors = [
            ("div", "ms-article-content msnt-styled-content"),
            ("div", "ms-article__body"),
            ("div", "article-body"),
            ("article", ""),
        ]

        main = None
        for tag, class_name in possible_selectors:
            if class_name:
                main = soup.find(tag, class_=class_name)
            else:
                main = soup.find(tag)
            if main:
                break

        if main is None:
            all_paragraphs = soup.find_all("p")
            text_article = " ".join([p.text for p in all_paragraphs if len(p.text) > 50])
        else:
            sentences = main.find_all("p")
            text_article = " ".join([item.text for item in sentences])

        return text_article