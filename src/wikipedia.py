import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config import USER_AGENT

WIKI_API_TMPL = "https://{lang}.wikipedia.org/w/api.php"

HEADERS = {
    # Wikipedia recommends a descriptive User-Agent
    "User-Agent": USER_AGENT,
    "Accept": "application/json"
}

# ~3,000-4,000 words beyond the lead, capped so prompt length (and therefore inference
# time) stays bounded and predictable across the sample regardless of how long any one
# person's article runs - a well-documented figure's full article can be tens of
# thousands of characters.
DEFAULT_MAX_CHARS = 24000

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=20))
def get_wikipedia_full_extract(title: str, lang: str = "en", max_chars: int = DEFAULT_MAX_CHARS) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title
    }

    r = requests.get(WIKI_API_TMPL.format(lang=lang), params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()

    data = r.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    extract = page.get("extract", "") or ""
    return extract[:max_chars]
