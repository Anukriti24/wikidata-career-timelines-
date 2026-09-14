import json
import sys
import time
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz
from tqdm import tqdm
from src.config import RAW_PANTHEON_PATH
from src.pantheon import _read_pantheon_csv
from src.johnny5_baseline import classify_qid

TIMELINES_PANTHEON_EN_PATH = Path("data/processed/timelines_pantheon_en.parquet")
TIMELINES_PANTHEON_DE_PATH = Path("data/processed/timelines_pantheon_de.parquet")
EVALUATION_PANTHEON_EN_PATH = Path("data/processed/evaluation_report_pantheon_en.json")
EVALUATION_PANTHEON_DE_PATH = Path("data/processed/evaluation_report_pantheon_de.json")
JOHNNY5_RESULTS_PANTHEON_PATH = Path("data/processed/johnny5_results_pantheon.parquet")
CHECKPOINT_EVERY = 25
JOHNNY5_MAX_RETRIES = 3
JOHNNY5_RETRY_DELAY = 2  # seconds
FUZZY_MATCH_THRESHOLD = 70  # rapidfuzz partial_ratio score (0-100) counted as "matches"

# Maps the final dataset's reader-friendly column names (data/final/famous_individuals_pantheon.*)
# back onto the names used internally / in the raw pipeline output (timelines_pantheon_*.parquet),
# so run_evaluation_pantheon() can be pointed at either file.
_FINAL_DATASET_COLUMNS = {
    "occupations_raw": "occ_raw",
    "occupations_clean": "occ_clean",
    "career_timeline": "stages",
}

def _load_records(path: Path) -> pd.DataFrame:
    df = pd.read_json(path) if path.suffix == ".json" else pd.read_parquet(path)
    return df.rename(columns={k: v for k, v in _FINAL_DATASET_COLUMNS.items() if k in df.columns})

def _safe_list(value) -> list:
    """Normalize a DataFrame cell that should be a list but may come back as None or
    NaN - pandas fills a column with NaN (not None) for any row where the source dict
    was simply missing that key, which happens for every error record (which only has
    qid/error, nothing else). NaN is neither None nor falsy, so naive `x or []` / `x is
    None` checks silently let it through and list(nan) crashes."""
    if value is None or isinstance(value, float):
        return []
    return list(value)

def _safe_str(value) -> str:
    if value is None or isinstance(value, float):
        return ""
    return str(value)

def _matches_any(term: str, candidates: list[str], threshold: int = FUZZY_MATCH_THRESHOLD) -> bool:
    term = term.lower()
    return any(fuzz.partial_ratio(term, c.lower()) >= threshold for c in candidates if c)

def compare_to_pantheon(records: pd.DataFrame, pantheon: pd.DataFrame) -> pd.DataFrame:
    """
    For each person, check whether Pantheon's single broad occupation category is
    reflected (via fuzzy match) among the LLM's cleaned occupations, and whether the
    LLM produced a more granular set of labels than Pantheon's one broad category.
    """
    pantheon_lookup = pantheon.set_index("wd_id")["occupation"].to_dict()

    rows = []
    for _, r in records.iterrows():
        qid = r["qid"]
        broad = pantheon_lookup.get(qid)
        occ_clean = _safe_list(r.get("occ_clean"))

        if broad is None or pd.isna(broad) or not occ_clean:
            continue

        matched = _matches_any(broad, occ_clean)
        rows.append({
            "qid": qid,
            "pantheon_occupation": broad,
            "llm_occ_clean": occ_clean,
            "matches_pantheon": matched,
            "llm_more_specific": matched and len(occ_clean) > 1,
        })

    return pd.DataFrame(rows)

def check_internal_consistency(records: pd.DataFrame, pantheon: pd.DataFrame) -> pd.DataFrame:
    """
    Automated sanity checks (proposal section F, second approach):
    - stages are chronologically ordered (non-decreasing start_year)
    - inferred years fall within the person's [birthyear, deathyear] life span
    - each stage's cited evidence is actually attested in the biography lead text
    """
    life_lookup = pantheon.set_index("wd_id")[["birthyear", "deathyear"]].to_dict("index")

    rows = []
    for _, r in records.iterrows():
        qid = r["qid"]
        stages = _safe_list(r.get("stages"))
        bio = _safe_str(r.get("bio_lead"))
        life = life_lookup.get(qid, {})
        birth, death = life.get("birthyear"), life.get("deathyear")

        years = [s["start_year"] for s in stages if s.get("start_year") is not None]
        chronological = years == sorted(years)

        out_of_range = 0
        for s in stages:
            for y in (s.get("start_year"), s.get("end_year")):
                if y is None:
                    continue
                if birth is not None and y < birth:
                    out_of_range += 1
                elif death is not None and y > death:
                    out_of_range += 1

        unsupported = sum(
            1 for s in stages
            if s.get("evidence") and not _matches_any(s["evidence"], [bio])
        )

        rows.append({
            "qid": qid,
            "n_stages": len(stages),
            "chronological": chronological,
            "years_out_of_range": out_of_range,
            "stages_unsupported_by_bio": unsupported,
        })

    return pd.DataFrame(rows)

def _classify_qid_with_retries(qid: str) -> tuple[str | None, float | None, str | None]:
    """Attempt classify_qid up to JOHNNY5_MAX_RETRIES times with a short pause between
    attempts, to ride out transient rate-limiting/network blips instead of recording a
    permanent error on the first hiccup. Returns (label, prob_ratio, error)."""
    last_error = None
    for attempt in range(JOHNNY5_MAX_RETRIES):
        try:
            label, prob_ratio = classify_qid(qid)
            return label, prob_ratio, None
        except Exception as e:
            last_error = str(e)
            if attempt < JOHNNY5_MAX_RETRIES - 1:
                time.sleep(JOHNNY5_RETRY_DELAY)
    return None, None, last_error

def _load_johnny5_results(path: Path) -> dict:
    if not path.exists():
        return {}
    return {row["qid"]: row for row in pd.read_parquet(path).to_dict("records")}

def _save_johnny5_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(results.values())).to_parquet(path, index=False)

def run_evaluation_pantheon(
    timelines_path: Path = TIMELINES_PANTHEON_EN_PATH,
    pantheon_path: Path = RAW_PANTHEON_PATH,
    out_path: Path = EVALUATION_PANTHEON_EN_PATH,
    johnny5_results_path: Path = JOHNNY5_RESULTS_PANTHEON_PATH,
    lang: str = "en",
) -> dict:
    """
    lang="en" (default) runs the full evaluation. Any other lang skips the Pantheon
    occupation-match and Johnny5 comparisons (both English-only ground truths) and
    reports only the internal consistency checks - see module docstring.
    """
    records = _load_records(timelines_path)
    pantheon = _read_pantheon_csv(pantheon_path)
    consistency = check_internal_consistency(records, pantheon)

    if lang != "en":
        summary = {
            "lang": lang,
            "n_records": len(records),
            "pantheon_occupation_comparison": "not_applicable (English-only ground truth)",
            "johnny5_baseline": "not_applicable (English-only classifier)",
            "chronological_rate": float(consistency["chronological"].mean()) if len(consistency) else None,
            "records_with_out_of_range_years": int((consistency["years_out_of_range"] > 0).sum()) if len(consistency) else None,
            "records_with_unsupported_stages": int((consistency["stages_unsupported_by_bio"] > 0).sum()) if len(consistency) else None,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        return summary

    record_qids = set(records["qid"])
    pantheon_cmp = compare_to_pantheon(records, pantheon)

    johnny5_results = _load_johnny5_results(johnny5_results_path)

    # Only (re)attempt QIDs never classified before, or that previously errored -
    # already-successful classifications are reused as-is, same resumable pattern
    # as the main pipeline's process_people().
    todo = []
    for _, r in records.iterrows():
        qid = r["qid"]
        occ_clean = _safe_list(r.get("occ_clean"))
        if not occ_clean:
            continue
        existing = johnny5_results.get(qid)
        if existing is None or pd.notna(existing.get("error")):
            todo.append((qid, occ_clean))

    def _write_checkpoint(n_done: int) -> dict:
        # johnny5_results is a cache shared across every dataset ever evaluated with this
        # module, so it can hold far more QIDs than this run's records - restrict the
        # summary to just this dataset's QIDs instead of the whole accumulated cache.
        relevant_johnny5 = [v for qid, v in johnny5_results.items() if qid in record_qids]
        johnny5_cmp = pd.DataFrame(relevant_johnny5) if relevant_johnny5 else pd.DataFrame()
        johnny5_ok = johnny5_cmp[johnny5_cmp["error"].isna()] if len(johnny5_cmp) else johnny5_cmp

        summary = {
            "lang": lang,
            "n_records": len(records),
            "n_compared_to_pantheon": len(pantheon_cmp),
            "pantheon_match_rate": float(pantheon_cmp["matches_pantheon"].mean()) if len(pantheon_cmp) else None,
            "llm_more_specific_rate": float(pantheon_cmp["llm_more_specific"].mean()) if len(pantheon_cmp) else None,
            "johnny5_progress": f"{n_done}/{len(todo)} attempted this run, {len(johnny5_results)} total known",
            "n_compared_to_johnny5": len(johnny5_ok),
            "johnny5_errors": int(johnny5_cmp["error"].notna().sum()) if len(johnny5_cmp) else None,
            "johnny5_match_rate": float(johnny5_ok["matches_johnny5"].mean()) if len(johnny5_ok) else None,
            "chronological_rate": float(consistency["chronological"].mean()) if len(consistency) else None,
            "records_with_out_of_range_years": int((consistency["years_out_of_range"] > 0).sum()) if len(consistency) else None,
            "records_with_unsupported_stages": int((consistency["stages_unsupported_by_bio"] > 0).sum()) if len(consistency) else None,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        return summary

    summary = _write_checkpoint(0)
    for i, (qid, occ_clean) in enumerate(tqdm(todo, desc="Johnny5 baseline (new/errored only)"), start=1):
        label, prob_ratio, error = _classify_qid_with_retries(qid)
        if error is None:
            johnny5_results[qid] = {
                "qid": qid, "johnny5_label": label, "johnny5_prob_ratio": prob_ratio,
                "matches_johnny5": _matches_any(label, occ_clean), "error": None,
            }
        else:
            johnny5_results[qid] = {"qid": qid, "johnny5_label": None, "johnny5_prob_ratio": None,
                                     "matches_johnny5": None, "error": error}

        if i % CHECKPOINT_EVERY == 0 or i == len(todo):
            _save_johnny5_results(johnny5_results, johnny5_results_path)
            summary = _write_checkpoint(i)

    return summary

def main():
    # optional: python -m src.evaluate_pantheon en|de  (default: both)
    langs = [sys.argv[1]] if len(sys.argv) > 1 else ["en", "de"]
    paths = {
        "en": (TIMELINES_PANTHEON_EN_PATH, EVALUATION_PANTHEON_EN_PATH),
        "de": (TIMELINES_PANTHEON_DE_PATH, EVALUATION_PANTHEON_DE_PATH),
    }
    for lang in langs:
        timelines_path, out_path = paths[lang]
        summary = run_evaluation_pantheon(timelines_path=timelines_path, out_path=out_path, lang=lang)
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
