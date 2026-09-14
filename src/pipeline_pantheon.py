from pathlib import Path
from src.timeline import process_people
from src.pantheon import sample_qids
from src.finalize_pantheon import build_final_dataset_pantheon, save_final_dataset_pantheon
from src.config import SAMPLE_SIZE, PIPELINE_MAX_WORKERS

TIMELINES_PANTHEON_EN_PATH = Path("data/processed/timelines_pantheon_en.parquet")
TIMELINES_PANTHEON_DE_PATH = Path("data/processed/timelines_pantheon_de.parquet")
FINAL_PANTHEON_JSON_PATH = Path("data/final/famous_individuals_pantheon.json")
FINAL_PANTHEON_PARQUET_PATH = Path("data/final/famous_individuals_pantheon.parquet")

def main():
    # Sampled once (requiring a German Wikipedia article, since Pantheon's CSV has no
    # wiki-edition-list column to check that from directly - see sample_qids) and reused
    # for both languages, so the bilingual final dataset covers the same people.
    sample = sample_qids(SAMPLE_SIZE, require_dewiki=True)

    process_people(sample, out_path=TIMELINES_PANTHEON_EN_PATH, lang="en", max_workers=PIPELINE_MAX_WORKERS, save_every=25)
    process_people(sample, out_path=TIMELINES_PANTHEON_DE_PATH, lang="de", max_workers=PIPELINE_MAX_WORKERS, save_every=25)

    df = build_final_dataset_pantheon(TIMELINES_PANTHEON_EN_PATH, TIMELINES_PANTHEON_DE_PATH)
    save_final_dataset_pantheon(df, FINAL_PANTHEON_JSON_PATH, FINAL_PANTHEON_PARQUET_PATH)
    print(f"Saved {len(df)} bilingual records")

if __name__ == "__main__":
    main()
