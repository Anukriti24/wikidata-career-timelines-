import pandas as pd
from pathlib import Path
from src.config import RAW_PANTHEON_PATH

KEEP_COLUMNS = ["wd_id", "name", "occupation", "birthyear", "deathyear", "hpi", "gender", "alive", "prob_ratio"]

def _read_pantheon_csv(path: Path = RAW_PANTHEON_PATH) -> pd.DataFrame:
    """
    Load the raw Pantheon CSV, keep only individuals (not groups) with a resolvable
    Wikidata QID, and trim to the columns useful for occupation-timeline work.
    """
    df = pd.read_csv(path)

    df = df[df["wd_id"].notna()]
    df = df[df["wd_id"].astype(str).str.startswith("Q")]
    df = df[df["is_group"] == False]

    keep = [c for c in KEEP_COLUMNS if c in df.columns]  # avoid crashing if a column is missing
    return df[keep]

def sample_qids(
    n: int,
    path: Path = RAW_PANTHEON_PATH,
    sort_by: str = "hpi",
    require_dewiki: bool = False,
    oversample_factor: int = 3,
) -> list[str]:
    """
    Return up to n unique Wikidata QIDs from the filtered Pantheon set, sorted
    by fame (hpi, descending) so the most well-documented biographies come first.

    Pantheon's own CSV has no per-person Wikipedia-edition-list column (unlike CVD's
    list_wikipedia_editions), so require_dewiki=True checks live against Wikidata
    instead: it walks the fame-sorted candidate list in oversample_factor*n-sized
    batches, keeping only QIDs with a German Wikipedia article, until n are collected
    (or candidates run out). Used to get a sample usable for a bilingual (en+de) run.
    """
    df = _read_pantheon_csv(path)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    candidates = df["wd_id"].dropna().unique().tolist()

    if not require_dewiki:
        return candidates[:n]

    from src.wikidata import fetch_wiki_titles

    selected: list[str] = []
    start = 0
    while len(selected) < n and start < len(candidates):
        batch = candidates[start: start + n * oversample_factor]
        start += len(batch)
        de_titles = fetch_wiki_titles(batch, lang="de")
        selected.extend(qid for qid in batch if de_titles.get(qid))

    return selected[:n]
