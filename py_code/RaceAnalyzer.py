from bs4 import BeautifulSoup
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from ArticleAnalyzer import ArticleAnalyzer
import time


class RaceAnalyzer(ArticleAnalyzer):
    def __init__(self, period=5):
        self.url = "https://www.the-race.com/sitemap-posts.xml"
        self.period = period
        super().__init__(self.url, self.period)

    def get_links(self):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        try:
            response = requests.get(self.url, headers=headers, timeout=30)
            response.raise_for_status()
        except:
            return []

        try:
            root = ET.fromstring(response.content)
        except:
            return []

        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        cutoff = datetime.now() - timedelta(days=self.period)
        articles = []

        for url_elem in root.findall('.//ns:url', namespaces=namespaces):
            loc = url_elem.find('ns:loc', namespaces=namespaces)
            lastmod = url_elem.find('ns:lastmod', namespaces=namespaces)
            if loc is None or loc.text is None:
                continue

            url = loc.text
            f1_patterns = ['/formula-1/', '/f1-', '/formula-one/']
            if not any(p in url.lower() for p in f1_patterns):
                continue

            exclude = ['/race-events/', '/promoted/', '/event-', '/tag/', '/category/']
            if any(p in url.lower() for p in exclude):
                continue

            date_str_formatted = ""
            if lastmod is not None and lastmod.text:
                try:
                    date_str = lastmod.text.strip()
                    if 'T' in date_str:
                        pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00')).replace(tzinfo=None)
                    else:
                        pub_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                    if pub_date >= cutoff:
                        date_str_formatted = pub_date.strftime('%Y-%m-%d')
                    else:
                        continue
                except:
                    pass

            articles.append({'url': url, 'title': "", 'date': date_str_formatted})

        return articles

    def get_text_by_link(self, single_link):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        try:
            response = requests.get(single_link, headers=headers, timeout=30)
            response.raise_for_status()
        except:
            return ""

        soup = BeautifulSoup(response.text, 'lxml')
        main = soup.find("div", class_="ms-article-content msnt-styled-content")

        if main is None:
            for tag, cls in [("div", "article-content"), ("div", "post-content"), ("div", "entry-content"), ("article", ""), ("main", "")]:
                main = soup.find(tag, class_=cls) if cls else soup.find(tag)
                if main:
                    break

        if main is None:
            return ""

        sentences = main.find_all("p")
        text_article = " ".join([item.text for item in sentences]).strip()
        time.sleep(0.5)
        return text_article