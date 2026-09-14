import re
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.wikidata import (
    fetch_person_occupations, fetch_wiki_title, fetch_people_occupations, fetch_wiki_titles,
    fetch_birth_death_years,
)
from src.wikipedia import get_wikipedia_full_extract
from src.llm_refine import build_timeline_with_llm

_QID_RE = re.compile(r"^Q\d+$")

def _person_label(person_occ: pd.DataFrame) -> str | None:
    labels = [l for l in person_occ["personLabel"].dropna().tolist() if not _QID_RE.match(l)]
    return labels[0] if labels else None

def _build_record(qid: str, occ_df: pd.DataFrame, title: str | None, life: dict | None = None, lang: str = "en") -> dict:
    person_occ = occ_df[occ_df["qid"] == qid] if not occ_df.empty else occ_df
    occ_labels = sorted({x for x in person_occ["occLabel"].dropna().tolist()})

    bio = get_wikipedia_full_extract(title, lang=lang) if title else ""

    life = life or {}
    timeline = build_timeline_with_llm(
        qid=qid, occupations=occ_labels, bio_lead=bio,
        birth_year=life.get("birth_year"), death_year=life.get("death_year"),
        lang=lang,
    )

    return {
        "qid": qid,
        "personLabel": _person_label(person_occ),
        "wiki_title": title,
        "bio_lead": bio,
        "occ_raw": occ_labels,
        "occ_clean": timeline.occupations_clean,
        "stages": [s.model_dump() for s in timeline.stages],
        "notes": timeline.notes,
    }

def process_one_person(qid: str, lang: str = "en") -> dict:
    occ_df = fetch_person_occupations(qid, lang=lang)
    title = fetch_wiki_title(qid, lang=lang)
    life = fetch_birth_death_years([qid]).get(qid)
    return _build_record(qid, occ_df, title, life, lang=lang)

def process_people(
    qids: list[str], out_path: Path | None = None, save_every: int = 1,
    lang: str = "en", max_workers: int = 1,
) -> list[dict]:
    existing: dict[str, dict] = {}
    if out_path is not None and out_path.exists():
        existing = {r["qid"]: r for r in pd.read_parquet(out_path).to_dict("records")}

    todo = [qid for qid in qids if qid not in existing or pd.notna(existing[qid].get("error"))]
    records = dict(existing)

    if todo:
        occ_df = fetch_people_occupations(todo, lang=lang)
        titles = fetch_wiki_titles(todo, lang=lang)
        life_years = fetch_birth_death_years(todo)

        def _process(qid: str) -> tuple[str, dict]:
            try:
                return qid, _build_record(qid, occ_df, titles.get(qid), life_years.get(qid), lang=lang)
            except Exception as e:
                return qid, {"qid": qid, "error": str(e)}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_process, qid) for qid in todo]
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f"Building timelines ({lang})")):
                qid, record = future.result()
                records[qid] = record

                if out_path is not None and (i + 1) % save_every == 0:
                    save_records(list(records.values()), out_path)

    if out_path is not None:
        save_records(list(records.values()), out_path)

    return [records[qid] for qid in qids]

def save_records(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
