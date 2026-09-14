import pandas as pd
from pathlib import Path
from src.cvd import RAW_CVD_PATH, _read_cvd_csv

TIMELINES_CVD_EN_PATH = Path("data/processed/timelines_cvd_en.parquet")
TIMELINES_CVD_DE_PATH = Path("data/processed/timelines_cvd_de.parquet")
FINAL_CVD_JSON_PATH = Path("data/final/famous_individuals_cvd.json")
FINAL_CVD_PARQUET_PATH = Path("data/final/famous_individuals_cvd.parquet")

def _lang_fields(row: pd.Series | None) -> dict:
    if row is None:
        return {"occ_raw": [], "occ_clean": [], "stages": [], "bio_lead": None}
    occ_raw = row.get("occ_raw")
    occ_clean = row.get("occ_clean")
    stages = row.get("stages")
    return {
        "occ_raw": list(occ_raw) if occ_raw is not None else [],
        "occ_clean": list(occ_clean) if occ_clean is not None else [],
        "stages": list(stages) if stages is not None else [],
        "bio_lead": row.get("bio_lead"),
    }

def build_final_dataset_cvd(
    timelines_en_path: Path = TIMELINES_CVD_EN_PATH,
    timelines_de_path: Path = TIMELINES_CVD_DE_PATH,
    cvd_path: Path = RAW_CVD_PATH,
) -> pd.DataFrame:
    en = pd.read_parquet(timelines_en_path)
    en = en[en["error"].isna() if "error" in en.columns else slice(None)].set_index("qid")
    de = pd.read_parquet(timelines_de_path)
    de = de[de["error"].isna() if "error" in de.columns else slice(None)].set_index("qid")

    cvd = _read_cvd_csv(cvd_path, ["wikidata_code", "name", "birth", "death"])
    life_lookup = cvd.set_index("wikidata_code")[["name", "birth", "death"]].to_dict("index")

    rows = []
    for qid in sorted(set(en.index) | set(de.index)):
        en_r = en.loc[qid] if qid in en.index else None
        de_r = de.loc[qid] if qid in de.index else None
        if en_r is None and de_r is None:
            continue  # both languages failed for this person; nothing to include

        life = life_lookup.get(qid, {})
        en_fields = _lang_fields(en_r)
        de_fields = _lang_fields(de_r)

        rows.append({
            "qid": qid,
            "name": life.get("name") or (en_r if en_r is not None else de_r).get("personLabel"),
            "birthyear": life.get("birth"),
            "deathyear": life.get("death"),
            "occupations_raw": {"en": en_fields["occ_raw"], "de": de_fields["occ_raw"]},
            "occupations_clean": {"en": en_fields["occ_clean"], "de": de_fields["occ_clean"]},
            "career_timeline": {"en": en_fields["stages"], "de": de_fields["stages"]},
            "bio_lead": {"en": en_fields["bio_lead"], "de": de_fields["bio_lead"]},
        })

    return pd.DataFrame(rows)

def save_final_dataset_cvd(
    df: pd.DataFrame,
    json_path: Path = FINAL_CVD_JSON_PATH,
    parquet_path: Path = FINAL_CVD_PARQUET_PATH,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        # The nested dict columns (bio_lead/occupations_*/career_timeline) can occasionally
        # defeat pyarrow's struct-schema inference (e.g. from all-empty lists in some rows) -
        # fall back to JSON-encoding just those columns for the parquet copy. The JSON output
        # above is unaffected either way since pandas serializes nested values natively there.
        import json
        flat = df.copy()
        for col in ("occupations_raw", "occupations_clean", "career_timeline", "bio_lead"):
            flat[col] = flat[col].apply(json.dumps)
        flat.to_parquet(parquet_path, index=False)

def main():
    df = build_final_dataset_cvd()
    save_final_dataset_cvd(df)
    print(f"Saved {len(df)} records to {FINAL_CVD_JSON_PATH} and {FINAL_CVD_PARQUET_PATH}")

if __name__ == "__main__":
    main()
