"""Simple subject-based web crawler for educational RAG sources."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.parse import unquote

import requests

from core.document_loader import DocumentSection
from core.rag_indexer import RAGIndexer


DEFAULT_CRAWL_SOURCES = {
    "filosofia": [
        "https://pt.wikipedia.org/wiki/Arist%C3%B3teles",
        "https://pt.wikipedia.org/wiki/%C3%89tica_a_Nic%C3%B4maco",
    ],
    "matematica": [
        "https://pt.wikipedia.org/wiki/Matem%C3%A1tica",
        "https://pt.wikipedia.org/wiki/%C3%81lgebra",
    ],
    "historia": [
        "https://pt.wikipedia.org/wiki/Hist%C3%B3ria",
        "https://pt.wikipedia.org/wiki/Hist%C3%B3ria_de_Portugal",
    ],
    "ciencias": [
        "https://pt.wikipedia.org/wiki/Ci%C3%AAncia",
        "https://pt.wikipedia.org/wiki/Biologia",
    ],
}


@dataclass(frozen=True)
class CrawlSource:
    subject: str
    url: str
    max_pages: int = 2


class _ReadableHTMLParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "form"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(urljoin(self.base_url, href))
        if tag in {"p", "br", "li", "h1", "h2", "h3", "h4"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "form"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in {"p", "li", "h1", "h2", "h3", "h4"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += f" {text}"
        else:
            self.text_parts.append(text)

    def readable_text(self) -> str:
        text = " ".join(self.text_parts)
        text = re.sub(r"\s*\n\s*", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\[\s*\d+\s*\]", "", text)
        text = re.sub(r"\[\s*editar(?:\s*\|\s*editar código)?\s*\]", "", text, flags=re.IGNORECASE)
        text = re.split(
            r"\n(?:Referências|Bibliografia|Ligações externas|Ver também|Notas)\n",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return text.strip()


class WebCrawler:
    def __init__(self, indexer: RAGIndexer, timeout: int = 15):
        self.indexer = indexer
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "EduBotCrawler/1.0 (+local educational RAG project)",
                "Accept": "text/html,application/xhtml+xml",
            }
        )

    def crawl_subject(
        self,
        subject: str,
        urls: list[str],
        max_pages_per_seed: int = 2,
        progress=None,
    ) -> dict[str, Any]:
        sections: list[DocumentSection] = []
        visited: set[str] = set()

        for url in urls:
            for page in self._crawl_seed(url, max_pages_per_seed, visited, progress):
                section = DocumentSection(
                    source=page["title"] or page["url"],
                    text=page["text"],
                    metadata={
                        "source_type": "web",
                        "subject": subject,
                        "url": page["url"],
                        "title": page["title"] or page["url"],
                    },
                )
                sections.append(section)

        chunks = self.indexer.index_sections(
            sections,
            reset_where={"$and": [{"source_type": "web"}, {"subject": subject}]},
            extra_metadata={"source_type": "web", "subject": subject},
            progress=progress,
        )
        return {"subject": subject, "pages": len(sections), "chunks": chunks}

    def crawl_sources(self, sources: list[CrawlSource], progress=None) -> list[dict[str, Any]]:
        grouped: dict[str, list[CrawlSource]] = defaultdict(list)
        for source in sources:
            grouped[source.subject].append(source)

        results = []
        for subject, subject_sources in grouped.items():
            urls = [source.url for source in subject_sources]
            max_pages = max(source.max_pages for source in subject_sources)
            results.append(
                self.crawl_subject(
                    subject,
                    urls,
                    max_pages_per_seed=max_pages,
                    progress=progress,
                )
            )
        return results

    def _crawl_seed(self, seed_url: str, max_pages: int, global_visited: set[str], progress=None):
        seed_url = self._normalize_url(seed_url)
        seed_netloc = urlparse(seed_url).netloc
        queue = deque([seed_url])
        local_count = 0

        while queue and local_count < max_pages:
            url = queue.popleft()
            if url in global_visited:
                continue
            global_visited.add(url)

            page = self._fetch_page(url)
            if page is None:
                continue

            local_count += 1
            if progress:
                progress(f"Crawled {url}")
            yield page

            for link in page["links"]:
                clean_link = self._normalize_url(link)
                parsed = urlparse(clean_link)
                if parsed.scheme not in {"http", "https"}:
                    continue
                if parsed.netloc != seed_netloc:
                    continue
                if clean_link not in global_visited:
                    queue.append(clean_link)

    def _fetch_page(self, url: str) -> dict[str, Any] | None:
        if "wikipedia.org" in urlparse(url).netloc:
            wiki_page = self._fetch_wikipedia_page(url)
            if wiki_page is not None:
                return wiki_page

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            return None

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return None

        parser = _ReadableHTMLParser(url)
        parser.feed(response.text)
        text = parser.readable_text()
        if len(text) < 300:
            return None

        return {
            "url": url,
            "title": parser.title.strip() or url,
            "text": text,
            "links": parser.links,
        }

    def _fetch_wikipedia_page(self, url: str) -> dict[str, Any] | None:
        parsed = urlparse(url)
        title = unquote(parsed.path.rsplit("/", 1)[-1]).replace("_", " ")
        if not title:
            return None

        api_url = f"{parsed.scheme}://{parsed.netloc}/w/api.php"
        try:
            response = self.session.get(
                api_url,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": "1",
                    "redirects": "1",
                    "format": "json",
                    "titles": title,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return None

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        text = page.get("extract", "").strip()
        page_title = page.get("title", title)
        if len(text) < 300:
            return None

        text = re.split(
            r"\n=+\s*(Referências|Bibliografia|Ligações externas|Ver também|Notas)\s*=+\n",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        return {
            "url": url,
            "title": page_title,
            "text": text,
            "links": [],
        }

    @staticmethod
    def _normalize_url(url: str) -> str:
        clean, _fragment = urldefrag(url)
        return clean.rstrip("/")


def default_sources(max_pages: int = 2) -> list[CrawlSource]:
    sources = []
    for subject, urls in DEFAULT_CRAWL_SOURCES.items():
        for url in urls:
            sources.append(CrawlSource(subject=subject, url=url, max_pages=max_pages))
    return sources


def source_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()
