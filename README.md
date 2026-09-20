# Integrating LLM into Wikidata and Wikipedia

Bilingual (EN/DE) career timelines for famous people, built by combining Wikidata
occupations with Wikipedia biography text, then refined by a local LLM (Ollama,
llama3.1:8b) into dated career stages with evidence quotes. Two source samples -
Pantheon (fame-ranked) and the Cross-Verified Database (random sample) - each produce
their own bilingual dataset, browsable through a Streamlit app.

Fully reproducible from scratch: both source datasets are public downloads, every
processing step is a plain Python module in `src/`, and the pipeline checkpoints and
resumes on its own - nothing here depends on manual steps or state that only exists on
one machine.

## Reproducing the pipeline from scratch

Needs an Ollama server for the LLM refinement step, and takes a while - each dataset is
~10,000 people x 2 languages, expect several hours of inference even on a decent GPU.

### 1. Set up the environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Requires Python 3.11+.

### 2. Get an Ollama server running

Install Ollama (https://ollama.com) and pull the model:

```bash
ollama pull llama3.1:8b
ollama serve
```

By default the code looks for Ollama at `http://localhost:11434/api/generate`. If it's
running somewhere else (a remote GPU box, a different port), point at it with an env
var instead:

```bash
export OLLAMA_URL=http://<host>:<port>/api/generate
```

### 3. Get the raw source data

Neither raw dataset is committed here - they're large, and both are public downloads.
`data/raw/DOWNLOAD_LINKS.txt` has the exact source for each one and what to name the
file:

- Pantheon -> `data/raw/pantheon.csv`
- Cross-Verified Database -> `data/raw/cross_verified_database.csv`

Nothing needs pre-processing by hand - both pipelines read and filter their raw CSV
directly, every time, no separate prep step.

### 4. Run a pipeline

Each dataset is independent; run whichever you want (or both).

```bash
# Cross-Verified Database
.venv/bin/python -m src.pipeline_cvd
.venv/bin/python -m src.evaluate_cvd

# Pantheon
.venv/bin/python -m src.pipeline_pantheon
.venv/bin/python -m src.evaluate_pantheon
```

The pipeline step (`pipeline_cvd`/`pipeline_pantheon`) fetches occupations + biography
text for the sampled people, runs them through the LLM, and writes the bilingual final
dataset to `data/final/`. The evaluate step scores the result against two independent
baselines (the dataset's own occupation labels, and a separate text classifier) plus
some automated sanity checks on the generated timelines, and writes a report to
`data/processed/`.

Both steps checkpoint as they go (`data/processed/timelines_*.parquet`) and skip people
they've already succeeded on if you re-run them - kill the process partway through and
re-run the same command, and it picks up where it left off instead of starting over.
`SAMPLE_SIZE` and `PIPELINE_MAX_WORKERS` are env-var overridable (see `src/config.py`)
if you want to verify the whole chain works end to end before committing to a full run,
e.g. `SAMPLE_SIZE=30 .venv/bin/python -m src.pipeline_cvd`.

`data/final/*.json` is what `app.py` actually reads; the pipeline also writes a
`.parquet` copy of the same data next to it (for programmatic analysis outside this
project), but that copy isn't tracked in this repo since nothing here needs it back.
If you want it, regenerate it directly from the checkpointed `data/processed/`
output - no LLM/Ollama and no re-fetching needed, just a re-join:

```bash
.venv/bin/python -m src.finalize_cvd
.venv/bin/python -m src.finalize_pantheon
```

### 5. Browse the result

```bash
.venv/bin/streamlit run app.py
```

Applink - https://4fvzs5wrtxks7pz3cbimc6.streamlit.app/?view=search
Hosted by Streamlit
