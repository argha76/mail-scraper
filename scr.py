#!/usr/bin/env python3
"""
main.py — high-volume async SerpAPI email scraper with full logging.

Run example:
  python main.py --queries-file queries.txt --max-results 30 --max-per-site 25 --concurrency 50 --log-level DEBUG

Requirements:
  pip install aiohttp lxml tldextract
  # optional for MX validation:
  pip install aiodns
"""

from __future__ import annotations
import argparse
import asyncio
import csv
import json
import re
import time
import logging
from collections import defaultdict, deque
from urllib.parse import urlparse, urljoin
from typing import Optional

import aiohttp
import tldextract

from concurrent.futures import ThreadPoolExecutor
from lxml import html

# Optional async DNS
try:
    import aiodns
    AIODNS_AVAILABLE = True
except Exception:
    AIODNS_AVAILABLE = False

import urllib.robotparser

# ---------------- CONFIG / DEFAULTS ----------------
# Hardcoded SerpAPI key (override with --serpapi-key to be safer)
SERPAPI_KEY = "fedc7db3ab5bf4f17f6fb2d5e273367ef90103f13ab0d96e1e5801a595e0138b"

USER_AGENT_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121 Safari/537.36",
]

DEFAULT_TIMEOUT = 12
GLOBAL_CONCURRENCY = 50
PER_DOMAIN_CONCURRENCY = 6
ROBOTS_CACHE_TTL = 3600

# Expanded candidate paths
MAX_CANDIDATE_PATHS = [
    "/", "/contact", "/contact-us", "/contactus",
    "/about", "/about-us", "/aboutus",
    "/team", "/staff", "/people", "/directory",
    "/support", "/help", "/faq",
    "/careers", "/jobs",
    "/legal", "/impressum", "/privacy",
    "/press", "/media", "/company", "/company/contact",
    "/office", "/locations", "/our-team", "/team.html"
]

# Improved email regex
EMAIL_RE = re.compile(
    r"""
    (?:(?:[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+)*)|
       (?:"(?:\\.|[^"\\])*"))
    @
    (?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}
    """,
    re.VERBOSE | re.IGNORECASE
)

SKIP_DOMAINS_CONTAINS = {
    "google.", "bing.", "duckduckgo.", "serpapi.", "facebook.", "twitter.",
    "linkedin.", "youtube.", "wikipedia.", "medium.", "tumblr."
}

# ---------------- Logging (default DEBUG to ensure visibility) ----------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scraper")

# Immediate visible startup print
print("=== SCRAPER STARTED ===")

# ---------------- Utility functions ----------------
def normalize_domain(url: str) -> str:
    p = urlparse(url)
    host = p.netloc.lower()
    if ":" in host:
        host = host.split(":")[0]
    return host

def pick_user_agent(idx: int = 0) -> str:
    return USER_AGENT_POOL[idx % len(USER_AGENT_POOL)]

def deobfuscate_text(s: str) -> str:
    s = s.replace("[at]", "@").replace("(at)", "@").replace(" at ", "@").replace(" AT ", "@")
    s = s.replace("[dot]", ".").replace("(dot)", ".").replace(" dot ", ".").replace(" DOT ", ".")
    s = s.replace("mailto:", "")
    s = re.sub(r'\s*@\s*', '@', s)
    s = re.sub(r'\s*\.\s*', '.', s)
    return s

def extract_emails_from_text(text: str) -> set:
    if not text:
        return set()
    candidates = set()
    deob = deobfuscate_text(text)
    for m in EMAIL_RE.finditer(deob):
        candidates.add(m.group(0).lower())
    for m in EMAIL_RE.finditer(text):
        candidates.add(m.group(0).lower())
    filtered = set()
    for e in candidates:
        if any(e.endswith(ext) for ext in (".png", ".jpg", ".gif", ".svg", ".css", ".js")):
            continue
        if e.count("@") != 1:
            continue
        local, domain = e.split("@", 1)
        if len(local) < 1 or len(domain) < 3:
            continue
        filtered.add(e)
    return filtered

# ---------------- Robots cache ----------------
class RobotsCache:
    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
        self._cache = {}

    async def allows(self, url: str, user_agent: str = "*") -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        now = time.time()
        entry = self._cache.get(base)
        if entry and now - entry[0] < ROBOTS_CACHE_TTL:
            rp = entry[1]
            allowed = rp.can_fetch(user_agent, url)
            logger.debug("robots cache for %s -> %s", base, allowed)
            return allowed
        robots_url = base.rstrip("/") + "/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        try:
            async with self._session.get(robots_url, timeout=DEFAULT_TIMEOUT) as resp:
                if resp.status == 200:
                    txt = await resp.text()
                    rp.parse(txt.splitlines())
                    logger.debug("fetched robots.txt for %s (len=%d)", base, len(txt))
                else:
                    rp.parse([])
                    logger.debug("robots.txt not found (%s): status=%s", robots_url, resp.status)
        except Exception as e:
            logger.debug("robots fetch error %s: %s", robots_url, e)
            rp.parse([])
        self._cache[base] = (now, rp)
        allowed = rp.can_fetch(user_agent, url)
        logger.debug("robots final for %s -> %s", base, allowed)
        return allowed

# ---------------- MX checker ----------------
class MXChecker:
    def __init__(self):
        self._resolver = aiodns.DNSResolver() if AIODNS_AVAILABLE else None
        self._cache = {}

    async def has_mx(self, domain: str) -> bool:
        domain = domain.lower()
        if domain in self._cache:
            logger.debug("mx cache hit %s -> %s", domain, self._cache[domain])
            return self._cache[domain]
        if not self._resolver:
            self._cache[domain] = True
            return True
        try:
            answers = await self._resolver.query(domain, 'MX')
            ok = len(answers) > 0
        except Exception as e:
            logger.debug("mx query error for %s: %s", domain, e)
            ok = False
        self._cache[domain] = ok
        logger.debug("mx result %s -> %s", domain, ok)
        return ok

# ---------------- Async Scraper ----------------
class Scraper:
    def __init__(self, serpapi_key: str,
                 concurrency: int = GLOBAL_CONCURRENCY,
                 per_domain: int = PER_DOMAIN_CONCURRENCY,
                 mx_validate: bool = False,
                 max_per_site: int = 25,
                 max_pages_per_site: int = 12,
                 pages_per_domain: int = 1,
                 timeout: int = DEFAULT_TIMEOUT):
        self.serpapi_key = serpapi_key
        self.max_per_site = max_per_site
        self.max_pages_per_site = max_pages_per_site
        self.pages_per_domain = pages_per_domain
        self.mx_validate = mx_validate and AIODNS_AVAILABLE
        self.timeout = timeout
        self.sem = asyncio.Semaphore(concurrency)
        self.domain_locks = defaultdict(lambda: asyncio.Semaphore(per_domain))
        self.executor = ThreadPoolExecutor(max_workers=12)
        self.session = aiohttp.ClientSession(headers={"User-Agent": pick_user_agent()})
        self.robots = RobotsCache(self.session)
        self.mxchecker = MXChecker() if self.mx_validate else None
        self.results = []
        self._domain_page_count = defaultdict(int)

    async def close(self):
        await self.session.close()
        self.executor.shutdown(wait=False)
        logger.debug("session closed and executor shutdown")

    async def serpapi_search(self, query: str, max_results: int = 30) -> list:
        logger.info("SerpAPI request for query: %s (max_results=%d)", query, max_results)
        url = "https://serpapi.com/search"
        params = {"q": query, "api_key": self.serpapi_key, "engine": "google", "num": max_results}
        async with self.sem:
            try:
                async with self.session.get(url, params=params, timeout=self.timeout) as resp:
                    logger.debug("SerpAPI HTTP %s for query %s", resp.status, query)
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error("SerpAPI non-200: %s %s", resp.status, text[:200])
                        raise RuntimeError(f"SerpAPI status {resp.status}")
                    data = await resp.json()
            except Exception as e:
                logger.exception("SerpAPI request failed for query '%s': %s", query, e)
                raise RuntimeError(f"SerpAPI request failed: {e}")
        urls = []
        for item in data.get("organic_results", []):
            link = item.get("link") or item.get("url")
            if not link:
                continue
            dom = normalize_domain(link)
            if any(skip in dom for skip in SKIP_DOMAINS_CONTAINS):
                logger.debug("Skipping search result (skip-domain) %s", link)
                continue
            if link not in urls:
                urls.append(link)
            if len(urls) >= max_results:
                break
        logger.info("SerpAPI returned %d usable links for query '%s'", len(urls), query)
        return urls

    async def fetch_page(self, url: str, used_ua_index: int = 0, proxy: Optional[str] = None) -> Optional[str]:
        dom = normalize_domain(url)
        if any(skip in dom for skip in SKIP_DOMAINS_CONTAINS):
            logger.debug("fetch_page skipping domain due to skip-list: %s", dom)
            return None
        allowed = await self.robots.allows(url, "*")
        if not allowed:
            logger.info("robots.txt disallows %s — skipping", url)
            return None
        async with self.domain_locks[dom]:
            headers = {"User-Agent": pick_user_agent(used_ua_index)}
            try:
                async with self.sem:
                    logger.debug("GET %s (UA=%s) proxy=%s", url, headers["User-Agent"], bool(proxy))
                    async with self.session.get(url, headers=headers, timeout=self.timeout, proxy=proxy) as resp:
                        logger.debug("HTTP %s for %s", resp.status, url)
                        if resp.status != 200:
                            return None
                        ctype = resp.headers.get("Content-Type", "")
                        if "text/html" in ctype or "text/plain" in ctype:
                            body = await resp.text()
                            logger.debug("fetched %d bytes from %s", len(body), url)
                            return body
                        logger.debug("unsupported content-type %s for %s", ctype, url)
                        return None
            except Exception as e:
                logger.debug("fetch error for %s: %s", url, e)
                return None

    async def parse_and_extract(self, html_text: str, base_url: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._parse_sync, html_text, base_url)

    def _parse_sync(self, html_text: str, base_url: str):
        out = []
        internal_links = []
        try:
            doc = html.fromstring(html_text)
        except Exception as e:
            logger.debug("lxml parse failed for %s: %s", base_url, e)
            text = re.sub(r'\s+', ' ', html_text)
            emails = extract_emails_from_text(text)
            for e in emails:
                out.append({"email": e, "source_url": base_url, "snippet": ""})
            return out, internal_links
        # mailto anchors
        try:
            for a in doc.xpath("//a[starts-with(translate(@href,'MAILTO','mailto'),'mailto:')]"):
                href = a.get("href", "")
                addr = href.split(":", 1)[-1].split("?")[0].strip()
                addr = deobfuscate_text(addr).lower()
                if EMAIL_RE.search(addr):
                    snippet = (a.text_content() or "").strip()[:200]
                    out.append({"email": addr, "source_url": base_url, "snippet": snippet})
        except Exception as e:
            logger.debug("mailto extraction error for %s: %s", base_url, e)
        # inline text
        try:
            text = doc.text_content()
            text_deob = deobfuscate_text(text)
            emails = set()
            for m in EMAIL_RE.finditer(text_deob):
                emails.add(m.group(0).lower())
            for m in EMAIL_RE.finditer(text):
                emails.add(m.group(0).lower())
            for e in emails:
                idx = text.lower().find(e.lower())
                snippet = ""
                if idx != -1:
                    start = max(0, idx - 80)
                    end = min(len(text), idx + 80)
                    snippet = text[start:end].strip().replace("\n", " ")
                out.append({"email": e, "source_url": base_url, "snippet": snippet})
        except Exception as e:
            logger.debug("inline email extraction error for %s: %s", base_url, e)
        # internal links
        try:
            for el in doc.xpath("//a[@href]"):
                href = el.get("href").strip()
                if href.startswith("http") or href.startswith("//"):
                    full = href if href.startswith("http") else "http:" + href
                else:
                    full = urljoin(base_url, href)
                try:
                    if normalize_domain(full) == normalize_domain(base_url):
                        internal_links.append(full)
                except Exception:
                    continue
        except Exception as e:
            logger.debug("internal links extraction error for %s: %s", base_url, e)
        logger.debug("parse %s -> emails=%d internal_links=%d", base_url, len(out), len(internal_links))
        return out, internal_links

    async def scrape_site(self, start_url: str, proxies: Optional[list] = None):
        logger.info("SCRAPING SITE: %s", start_url)
        parsed = urlparse(start_url)
        scheme = parsed.scheme if parsed.scheme else "http"
        root = f"{scheme}://{parsed.netloc}"
        collected = {}
        proxy = None
        if proxies:
            proxy = proxies[int(time.time() * 1000) % len(proxies)]
        # page queue
        q = deque()
        q.append(start_url)
        for p in MAX_CANDIDATE_PATHS:
            url = root.rstrip("/") + p if p.startswith("/") or p.startswith("#") else urljoin(root, p)
            if url not in q:
                q.append(url)
        visited = set()
        pages_seen = 0
        domain = normalize_domain(root)
        per_domain_allowed = self.pages_per_domain
        if self._domain_page_count[domain] >= per_domain_allowed:
            logger.info("pages-per-domain limit reached for %s", domain)
            return 0
        self._domain_page_count[domain] += 1

        while q and pages_seen < self.max_pages_per_site and len(collected) < self.max_per_site:
            candidate = q.popleft()
            if candidate in visited:
                continue
            visited.add(candidate)
            logger.debug("visiting %s (pages_seen=%d collected=%d)", candidate, pages_seen, len(collected))
            html_text = await self.fetch_page(candidate, proxy=proxy)
            if not html_text:
                logger.debug("no html returned for %s", candidate)
                continue
            pages_seen += 1
            parsed_results = await self.parse_and_extract(html_text, candidate)
            if isinstance(parsed_results, tuple):
                items, internal_links = parsed_results
            else:
                items, internal_links = parsed_results, []
            logger.debug("parsed %s -> found_items=%d internal_links=%d", candidate, len(items), len(internal_links))
            for item in items:
                email = item.get("email")
                if not email:
                    continue
                if self.mxchecker:
                    domain_e = email.split("@", 1)[1]
                    ok = await self.mxchecker.has_mx(domain_e)
                    if not ok:
                        logger.debug("mx check failed for %s, skipping", email)
                        continue
                if email not in collected:
                    collected[email] = item
                    logger.info("collected email %s from %s", email, candidate)
                if len(collected) >= self.max_per_site:
                    break
            # enqueue internal links
            for link in internal_links:
                if link not in visited and (len(q) + pages_seen) < (self.max_pages_per_site * 3):
                    q.append(link)
            await asyncio.sleep(0.06)
        logger.info("FINISHED %s | emails=%d pages=%d", start_url, len(collected), pages_seen)
        for e, info in collected.items():
            dom = normalize_domain(info["source_url"])
            self.results.append({"email": e, "source_url": info["source_url"], "domain": dom, "snippet": info.get("snippet","")})
        return len(collected)

    async def run_queries(self, queries: list, max_results: int = 30, proxies: Optional[list] = None):
        all_sites = []
        for q in queries:
            try:
                urls = await self.serpapi_search(q, max_results=max_results)
                logger.info("Query '%s' -> %d results", q, len(urls))
            except Exception as e:
                logger.error("SerpAPI error for '%s': %s", q, e)
                urls = []
            all_sites.extend(urls)
            await asyncio.sleep(0.2)
        unique = []
        seen = set()
        for u in all_sites:
            dom = normalize_domain(u)
            if dom in seen or any(skip in dom for skip in SKIP_DOMAINS_CONTAINS):
                logger.debug("skipping domain during dedupe/skip: %s", u)
                continue
            seen.add(dom)
            unique.append(u)
        logger.info("Total unique sites to scrape: %d", len(unique))
        tasks = []
        for u in unique:
            tasks.append(self.scrape_site(u, proxies=proxies))
        CHUNK = 128
        for i in range(0, len(tasks), CHUNK):
            chunk = tasks[i:i+CHUNK]
            logger.debug("gathering chunk %d..%d", i, i+len(chunk))
            await asyncio.gather(*chunk)
        return unique

# ---------------- Output helpers ----------------
def write_outputs(results: list, csv_file: str, jsonl_file: str):
    seen = set()
    rows = []
    for r in results:
        key = (r["email"], r["domain"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["email", "source_url", "domain", "snippet"])
        for r in rows:
            w.writerow([r["email"], r["source_url"], r["domain"], r.get("snippet","")])
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d deduped records to CSV/JSONL", len(rows))

# ---------------- CLI runner ----------------
async def main_async(args):
    # Adjust log level from CLI
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        logger.warning("Invalid log level %s, defaulting to DEBUG", args.log_level)
        numeric_level = logging.DEBUG
    logger.setLevel(numeric_level)
    logging.getLogger("aiohttp").setLevel(numeric_level)

    serp_key = args.serpapi_key or SERPAPI_KEY
    if not serp_key:
        logger.error("SerpAPI key required. Use --serpapi-key or set SERPAPI_KEY.")
        return 1

    queries = ["engineer in usa contact"]
    if args.query:
        queries.append(args.query)
    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    queries.append(ln)
    if not queries:
        logger.error("No queries provided. Use --query or --queries-file.")
        return 1

    proxies = None
    if args.proxies_file:
        with open(args.proxies_file, "r", encoding="utf-8") as fh:
            proxies = [ln.strip() for ln in fh if ln.strip()]
        logger.info("Loaded %d proxies", len(proxies))

    scraper = Scraper(
        serpapi_key=serp_key,
        concurrency=args.concurrency,
        per_domain=args.per_domain,
        mx_validate=args.mx_validate,
        max_per_site=args.max_per_site,
        max_pages_per_site=args.max_pages_per_site,
        pages_per_domain=args.pages_per_domain,
        timeout=args.timeout
    )

    try:
        start = time.time()
        unique = await scraper.run_queries(queries, max_results=args.max_results, proxies=proxies)
        elapsed = time.time() - start
        logger.info("Scraped %d unique sites (elapsed %.1f s).", len(unique), elapsed)
        write_outputs(scraper.results, args.output_csv, args.output_json)
        logger.info("Done: %d unique emails written to %s and %s", len(scraper.results), args.output_csv, args.output_json)
    finally:
        await scraper.close()
    return 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, help="Single query string")
    p.add_argument("--queries-file", type=str, help="File with one query per line")
    p.add_argument("--serpapi-key", type=str, help="SerpAPI key (overrides built-in)")
    p.add_argument("--max-results", type=int, default=30, help="SerpAPI results per query")
    p.add_argument("--concurrency", type=int, default=GLOBAL_CONCURRENCY, help="Global concurrent fetches")
    p.add_argument("--per-domain", type=int, default=PER_DOMAIN_CONCURRENCY, help="Parallel requests per domain")
    p.add_argument("--mx-validate", action="store_true", help="Enable MX validation (aiodns required)")
    p.add_argument("--proxies-file", type=str, help="Optional: file with proxies (one per line)")
    p.add_argument("--max-per-site", type=int, default=25, help="Max emails to collect per site")
    p.add_argument("--max-pages-per-site", type=int, default=12, help="Max pages to visit per site (internal crawl)")
    p.add_argument("--pages-per-domain", type=int, default=1, help="How many start pages per domain to allow (1 = default)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout seconds")
    p.add_argument("--output-csv", type=str, default="serp_async_v2.csv", help="CSV output path")
    p.add_argument("--output-json", type=str, default="serp_async_v2.jsonl", help="JSONL output path")
    p.add_argument("--log-level", type=str, default="DEBUG", help="Log level: DEBUG, INFO, WARNING, ERROR")
    args = p.parse_args()

    return asyncio.run(main_async(args))

if __name__ == "__main__":
    raise SystemExit(main())
