import pandas as pd
from pathlib import Path

RAW_CVD_PATH = Path("data/raw/cross_verified_database.csv")

def _read_cvd_csv(path: Path, usecols: list[str]) -> pd.DataFrame:
    """The raw CVD file is legitimately UTF-8 encoded (confirmed by inspecting its raw
    bytes), but a small number of rows elsewhere in the file aren't valid UTF-8, which
    makes a plain encoding="utf-8" read crash outright. encoding="latin-1" was previously
    used to avoid that crash, but silently mangles every accented name into mojibake
    (e.g. "Salvador_Dalí" -> "Salvador_DalÃ­") since latin-1 never raises - it just
    misdecodes every non-ASCII UTF-8 byte sequence as two separate characters. Reading as
    utf-8 with errors="replace" gets the correct decoding for the vast majority of rows
    and only degrades the handful of genuinely-bad ones (replaced with U+FFFD) instead of
    corrupting everything with accents. Centralized here so this fix can't regress in a
    third call site the way it already had in two others."""
    return pd.read_csv(path, usecols=usecols, encoding="utf-8", encoding_errors="replace")

def sample_qids_cvd(n: int, path: Path = RAW_CVD_PATH, sort_by: str = "ranking_visib_5criteria") -> list[str]:
    """
    Return up to n unique Wikidata QIDs from the Cross-Verified Database
    (Laouenan et al. 2022), sorted by fame rank ascending (rank 1 = most
    notable) so the most well-documented biographies come first.
    """
    df = _read_cvd_csv(path, ["wikidata_code", sort_by])
    df = df[df["wikidata_code"].notna()]
    df = df.sort_values(sort_by, ascending=True)
    return df["wikidata_code"].dropna().unique().tolist()[:n]

CVD_V2_SEED = 42

def sample_qids_cvd_v2(
    n: int,
    path: Path = RAW_CVD_PATH,
    birth_min: int = 1300,
    birth_max: int = 1950,
    min_wiki_editions: int = 2,
    require_editions: list[str] | None = ("enwiki", "dewiki"),
    seed: int = CVD_V2_SEED,
) -> list[str]:
    """
    Return up to n unique Wikidata QIDs randomly sampled (not fame-sorted) from
    the Cross-Verified Database, restricted to a subuniverse: born within
    [birth_min, birth_max], with at least min_wiki_editions Wikipedia language
    editions, and (if require_editions is set) specifically having every edition
    listed there - the pipeline builds each timeline from a specific language's
    Wikipedia lead paragraph, so an individual missing that edition would
    otherwise be sampled and then processed with no bio text at all. Defaults to
    requiring both English and German so the same sampled QIDs are usable for a
    bilingual (en+de) run.

    seed is fixed by default so the sample is reproducible across reruns.
    """
    df = _read_cvd_csv(path, ["wikidata_code", "birth", "number_wiki_editions", "list_wikipedia_editions"])
    df = df[df["wikidata_code"].notna()]
    df = df.drop_duplicates(subset="wikidata_code")
    df = df[df["birth"].between(birth_min, birth_max)]
    df = df[df["number_wiki_editions"] >= min_wiki_editions]
    for edition in (require_editions or []):
        df = df[df["list_wikipedia_editions"].str.contains(edition, na=False)]

    return df["wikidata_code"].sample(n=min(n, len(df)), random_state=seed).tolist()
