from pathlib import Path
from src.timeline import process_people
from src.cvd import sample_qids_cvd_v2
from src.finalize_cvd import build_final_dataset_cvd, save_final_dataset_cvd
from src.config import SAMPLE_SIZE, PIPELINE_MAX_WORKERS

TIMELINES_CVD_EN_PATH = Path("data/processed/timelines_cvd_en.parquet")
TIMELINES_CVD_DE_PATH = Path("data/processed/timelines_cvd_de.parquet")
FINAL_CVD_JSON_PATH = Path("data/final/famous_individuals_cvd.json")
FINAL_CVD_PARQUET_PATH = Path("data/final/famous_individuals_cvd.parquet")

def main():
    sample = sample_qids_cvd_v2(SAMPLE_SIZE)

    process_people(sample, out_path=TIMELINES_CVD_EN_PATH, lang="en", max_workers=PIPELINE_MAX_WORKERS, save_every=25)
    process_people(sample, out_path=TIMELINES_CVD_DE_PATH, lang="de", max_workers=PIPELINE_MAX_WORKERS, save_every=25)

    df = build_final_dataset_cvd(TIMELINES_CVD_EN_PATH, TIMELINES_CVD_DE_PATH)
    save_final_dataset_cvd(df, FINAL_CVD_JSON_PATH, FINAL_CVD_PARQUET_PATH)
    print(f"Saved {len(df)} bilingual records")

if __name__ == "__main__":
    main()
