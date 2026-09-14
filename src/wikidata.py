import re
import requests
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from urllib.parse import unquote
from src.config import USER_AGENT

WDQS_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": USER_AGENT
}

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=20))
def sparql(query: str) -> dict:
    r = requests.get(
        WDQS_URL,
        params={"query": query, "format": "json"},
        headers=HEADERS,
        timeout=60
    )
    r.raise_for_status()
    return r.json()

def fetch_people_occupations(qids: list[str], batch_size: int = 50, lang: str = "en") -> pd.DataFrame:
    """
    Fetch occupations (P106) for many people at once, batching QIDs into groups of
    batch_size per SPARQL request instead of one request per person. Includes optional
    start/end qualifiers (P580/P582).

    lang selects the label language (e.g. "de" for German personLabel/occLabel), with
    "en" always appended as a fallback so an occupation QID lacking a label in `lang`
    still resolves to something rather than None - a handful of occLabel values in a
    non-English pull may therefore legitimately come back in English.

    Returns a DataFrame with columns: qid, personLabel, occQid, occLabel, start, end.
    """
    all_rows = []
    label_langs = lang if lang == "en" else f"{lang},en"

    for i in range(0, len(qids), batch_size):
        chunk = qids[i:i + batch_size]
        values = " ".join(f"wd:{qid}" for qid in chunk)
        query = f"""
        SELECT ?person ?personLabel ?occ ?occLabel ?start ?end WHERE {{
          VALUES ?person {{ {values} }}
          OPTIONAL {{
            ?person p:P106 ?occStmt .
            ?occStmt ps:P106 ?occ .
            OPTIONAL {{ ?occStmt pq:P580 ?start . }}
            OPTIONAL {{ ?occStmt pq:P582 ?end . }}
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{label_langs}". }}
        }}
        """

        data = sparql(query)

        for b in data["results"]["bindings"]:
            all_rows.append({
                "qid": b["person"]["value"].rsplit("/", 1)[-1],
                "personLabel": b.get("personLabel", {}).get("value"),
                "occQid": b.get("occ", {}).get("value", "").split("/")[-1] if "occ" in b else None,
                "occLabel": b.get("occLabel", {}).get("value") if "occLabel" in b else None,
                "start": b.get("start", {}).get("value") if "start" in b else None,
                "end": b.get("end", {}).get("value") if "end" in b else None,
            })

    df = pd.DataFrame(all_rows)

    # Optional: drop empty occupation rows (in case Wikidata returns blanks)
    if not df.empty:
        df = df[df["occQid"].notna()]

    return df

def fetch_person_occupations(qid: str, lang: str = "en") -> pd.DataFrame:
    """Fetch occupations for a single person. See fetch_people_occupations to batch many at once."""
    return fetch_people_occupations([qid], lang=lang)

def fetch_wiki_titles(qids: list[str], batch_size: int = 50, lang: str = "en") -> dict:
    """
    Fetch the Wikipedia page titles (in the given language edition) linked to many
    Wikidata QIDs (via sitelinks), batching QIDs into groups of batch_size per SPARQL
    request. Returns {qid: title or None}.
    """
    titles = {qid: None for qid in qids}

    for i in range(0, len(qids), batch_size):
        chunk = qids[i:i + batch_size]
        values = " ".join(f"wd:{qid}" for qid in chunk)
        query = f"""
        SELECT ?person ?article WHERE {{
          VALUES ?person {{ {values} }}
          ?article schema:about ?person ;
                   schema:isPartOf <https://{lang}.wikipedia.org/> .
        }}
        """

        data = sparql(query)

        for b in data["results"]["bindings"]:
            qid = b["person"]["value"].rsplit("/", 1)[-1]
            url = b["article"]["value"]
            # Wikidata returns the article URL with the title percent-encoded (e.g.
            # "Udri%C8%99te" for "Udriște") - unquote before use, or any non-ASCII
            # title silently fails to resolve against Wikipedia's API later.
            titles[qid] = unquote(url.rsplit("/", 1)[-1]).replace("_", " ")

    return titles

def fetch_wiki_title(qid: str, lang: str = "en") -> str | None:
    """Fetch the Wikipedia title (in the given language edition) for a single QID. See
    fetch_wiki_titles to batch many at once."""
    return fetch_wiki_titles([qid], lang=lang)[qid]

def _parse_wikidata_year(date_str: str | None) -> int | None:
    """Wikidata returns dates as ISO 8601 (e.g. "1806-01-27T00:00:00Z", or
    "-0051-01-01T00:00:00Z" for 51 BCE) - extract just the (possibly negative) year."""
    if not date_str:
        return None
    m = re.match(r"^(-?\d+)-\d{2}-\d{2}", date_str)
    return int(m.group(1)) if m else None

def fetch_birth_death_years(qids: list[str], batch_size: int = 50) -> dict:
    """
    Fetch birth year (P569) and death year (P570) for many people at once, batching
    QIDs into groups of batch_size per SPARQL request. Returns
    {qid: {"birth_year": int|None, "death_year": int|None}}.

    Used to let the LLM-refinement guardrail (src/llm_refine.py) reject career-stage
    years that fall outside a person's actual lifespan or exactly match their own
    birth/death year (a specific, repeated model hallucination) - fetched from
    Wikidata directly, rather than from whichever raw dataset (Pantheon/CVD) supplied
    the QID, so this works uniformly for either pipeline track.
    """
    years = {qid: {"birth_year": None, "death_year": None} for qid in qids}

    for i in range(0, len(qids), batch_size):
        chunk = qids[i:i + batch_size]
        values = " ".join(f"wd:{qid}" for qid in chunk)
        query = f"""
        SELECT ?person ?birth ?death WHERE {{
          VALUES ?person {{ {values} }}
          OPTIONAL {{ ?person wdt:P569 ?birth . }}
          OPTIONAL {{ ?person wdt:P570 ?death . }}
        }}
        """

        data = sparql(query)

        for b in data["results"]["bindings"]:
            qid = b["person"]["value"].rsplit("/", 1)[-1]
            birth = _parse_wikidata_year(b.get("birth", {}).get("value"))
            death = _parse_wikidata_year(b.get("death", {}).get("value"))
            # only ever overwrite with a real value - a person with multiple
            # birth/death statements can produce more than one binding, and a later
            # one shouldn't clobber an already-found year with None
            if birth is not None:
                years[qid]["birth_year"] = birth
            if death is not None:
                years[qid]["death_year"] = death

    return years
