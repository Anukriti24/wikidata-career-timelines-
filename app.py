import html
import os
import tempfile
import threading
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from rapidfuzz import fuzz, process

DATA_DIR = Path(__file__).parent / "data" / "final"

# The two final datasets are too large for a normal git commit (100MB+), so they're
# published as GitHub Release assets instead and pulled down here on first run.
# See .gitignore (data/final/*.json) and the repo's Releases page.
_RELEASE_BASE = "https://github.com/Anukriti24/wikidata-career-timelines-/releases/download/data-v1"
DATASET_URLS = {
    "famous_individuals_pantheon.json": f"{_RELEASE_BASE}/famous_individuals_pantheon.json",
    "famous_individuals_cvd.json": f"{_RELEASE_BASE}/famous_individuals_cvd.json",
}

# Streamlit runs every user session as a thread inside one process, so two sessions
# can both see a dataset missing and race to download it. This lock serializes
# ensure_datasets() across those threads; each download also uses a unique temp
# filename (mkstemp) so even a missed lock can't cause two threads to fight over
# the same partial file.
_download_lock = threading.Lock()


def ensure_datasets() -> None:
    """Download any dataset files missing locally (fresh Streamlit Cloud containers
    start with an empty data/final/) from the GitHub Release that hosts them."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _download_lock:
        for filename, url in DATASET_URLS.items():
            dest = DATA_DIR / filename
            if dest.exists():
                continue
            with st.spinner(f"Downloading {filename} (one-time)..."):
                fd, tmp_name = tempfile.mkstemp(dir=DATA_DIR, prefix=f"{filename}.", suffix=".part")
                tmp = Path(tmp_name)
                try:
                    with requests.get(url, stream=True, timeout=60) as resp:
                        resp.raise_for_status()
                        with os.fdopen(fd, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                f.write(chunk)
                    tmp.rename(dest)
                except Exception:
                    tmp.unlink(missing_ok=True)
                    raise

# Friendly names shown in the dataset picker instead of raw filenames. Any dataset file
# that shows up without an entry here falls back to a title-cased version of its stem.
DATASET_TITLES = {
    "famous_individuals_pantheon.json": "Pantheon (fame-ranked)",
    "famous_individuals_cvd.json": "Cross-Verified Database",
}

LANGUAGE_LABELS = {"en": "English", "de": "German"}

# Columns that are nested per-language ({"en": ..., "de": ...}) in the bilingual final
# datasets - flattened to the selected language by select_lang() before rendering.
_BILINGUAL_COLUMNS = ["bio_lead", "occupations_raw", "occupations_clean", "career_timeline"]

st.set_page_config(page_title="Career Timelines", page_icon="🧭", layout="centered")

# Editorial/archival look (serif headings on warm paper tones) instead of default
# Streamlit chrome - .streamlit/config.toml sets the base palette so built-in widgets
# (sidebar, buttons, inputs) match; this adds the custom title + timeline components
# that aren't achievable through the theme config alone.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&display=swap');

.block-container { max-width: 760px; padding-top: 2.5rem; }

.app-title {
    font-family: 'Lora', Georgia, serif;
    font-size: 2.15rem;
    font-weight: 700;
    color: #2a2622;
    letter-spacing: -0.01em;
    margin-bottom: 0.2rem;
}
.app-subtitle { color: #6b6459; font-size: 0.95rem; margin-bottom: 2.2rem; }

.person-name {
    font-family: 'Lora', Georgia, serif;
    font-size: 2rem;
    font-weight: 600;
    color: #2a2622;
    margin-bottom: 0.15rem;
}
.person-meta { color: #6b6459; font-size: 0.9rem; margin-bottom: 0.9rem; }
.person-meta a { color: #a8441f; text-decoration: none; border-bottom: 1px solid #f3e3da; }
.person-meta a:hover { border-bottom-color: #a8441f; }

.occ-pill {
    display: inline-block;
    background: #f3e3da;
    color: #a8441f;
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0 0.35rem 0.4rem 0;
}

.timeline { position: relative; margin: 1.6rem 0 0.5rem 0; padding-left: 1.6rem; }
.timeline::before {
    content: "";
    position: absolute;
    left: 0.3rem;
    top: 0.4rem;
    bottom: 0.4rem;
    width: 2px;
    background: #e4dfd6;
}
.tl-stage { position: relative; padding-bottom: 1.6rem; }
.tl-stage:last-child { padding-bottom: 0; }
.tl-dot {
    position: absolute;
    left: -1.6rem;
    top: 0.3rem;
    width: 0.6rem;
    height: 0.6rem;
    border-radius: 50%;
    background: #a8441f;
    box-shadow: 0 0 0 3px #faf7f2;
}
.tl-dot.undated { background: #cdc6b8; }
.tl-year {
    font-size: 0.78rem;
    font-weight: 700;
    color: #a8441f;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.tl-year.undated { color: #9a9284; }
.tl-occ { font-family: 'Lora', Georgia, serif; font-size: 1.12rem; font-weight: 600; color: #2a2622; margin: 0.1rem 0 0.35rem 0; }
.tl-evidence {
    font-size: 0.92rem;
    color: #6b6459;
    line-height: 1.5;
    border-left: 2px solid #e4dfd6;
    padding-left: 0.85rem;
    font-style: italic;
}

.section-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: #9a9284;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 1.8rem 0 0.2rem 0;
}
</style>
"""


@st.cache_data(show_spinner=False)
def list_datasets() -> list[Path]:
    # Sorted by filename (not mtime, which shifts every time a dataset gets
    # regenerated) so the picker's default (index 0) stays stable - CVD sorts first.
    files = sorted(DATA_DIR.glob("famous_individuals*.json"), key=lambda p: p.name)
    return files


def dataset_title(path: Path) -> str:
    return DATASET_TITLES.get(path.name, path.stem.replace("_", " ").title())


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset(path_str: str) -> pd.DataFrame:
    df = pd.read_json(path_str)
    df["display_name"] = df["name"].str.replace("_", " ", regex=False).str.strip()
    return df


def select_lang(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Flatten the bilingual nested columns down to a single language, so the rest of
    the app (render_person, find_matches) can keep working with plain flat fields
    regardless of whether the loaded dataset is bilingual or (for any older/unmapped
    file) already flat."""
    out = df.copy()
    for col in _BILINGUAL_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = out[col].apply(lambda v: v.get(lang) if isinstance(v, dict) else v)
    return out


def format_years(birth, death) -> str:
    b = "?" if pd.isna(birth) else str(int(birth))
    if pd.isna(death):
        return f"b. {b}"
    return f"{b}–{int(death)}"


def format_stage_years(start, end) -> str:
    if pd.isna(start) and pd.isna(end):
        return "Year unknown"
    if pd.isna(start):
        return f"until {int(end)}"
    if pd.isna(end):
        return f"from {int(start)}"
    if int(start) == int(end):
        return str(int(start))
    return f"{int(start)}–{int(end)}"


def render_person(row: pd.Series) -> None:
    name = html.escape(row["display_name"])
    years = html.escape(format_years(row["birthyear"], row["deathyear"]))
    qid = html.escape(str(row["qid"]))

    occ_clean = row.get("occupations_clean") or []
    pills = "".join(f'<span class="occ-pill">{html.escape(o)}</span>' for o in occ_clean)

    st.markdown(
        f'<div class="person-name">{name}</div>'
        f'<div class="person-meta">{years} &nbsp;·&nbsp; '
        f'<a href="https://www.wikidata.org/wiki/{qid}" target="_blank">{qid}</a></div>'
        f'<div>{pills}</div>',
        unsafe_allow_html=True,
    )

    bio = row.get("bio_lead")
    if isinstance(bio, str) and bio.strip():
        with st.expander("Biography (Wikipedia lead)", expanded=False):
            st.write(bio)

    stages = row.get("career_timeline") or []
    if not len(stages):
        st.markdown('<div class="section-label">Career timeline</div>', unsafe_allow_html=True)
        st.caption("No career stages recorded for this person.")
        return

    st.markdown('<div class="section-label">Career timeline</div>', unsafe_allow_html=True)

    parts = ['<div class="timeline">']
    for stage in stages:
        undated = stage.get("start_year") is None and stage.get("end_year") is None
        dot_cls = "tl-dot undated" if undated else "tl-dot"
        year_cls = "tl-year undated" if undated else "tl-year"
        years_label = html.escape(format_stage_years(stage.get("start_year"), stage.get("end_year")))
        occ = html.escape(stage.get("occupation_clean") or "—")
        evidence = stage.get("evidence")

        parts.append('<div class="tl-stage">')
        parts.append(f'<div class="{dot_cls}"></div>')
        parts.append(f'<div class="{year_cls}">{years_label}</div>')
        parts.append(f'<div class="tl-occ">{occ}</div>')
        if evidence:
            parts.append(f'<div class="tl-evidence">&ldquo;{html.escape(evidence)}&rdquo;</div>')
        parts.append('</div>')
    parts.append('</div>')

    st.markdown("".join(parts), unsafe_allow_html=True)


def find_matches(df: pd.DataFrame, query: str, limit: int = 8):
    query_norm = query.strip().lower()
    if not query_norm:
        return df.iloc[0:0]

    lower_names = df["display_name"].str.lower()

    exact = df[lower_names == query_norm]
    if len(exact):
        return exact

    # Substring match (e.g. "einstein" -> "Albert Einstein") ranked by name length,
    # so the most specific / closest match comes first.
    substring = df[lower_names.str.contains(query_norm, regex=False)]
    if len(substring):
        order = lower_names.loc[substring.index].str.len().sort_values().index
        return substring.loc[order].head(limit)

    # Fall back to typo-tolerant fuzzy matching. token_sort_ratio (rather than
    # WRatio) avoids inflated scores on short/unrelated names for garbage queries.
    results = process.extract(query, df["display_name"], scorer=fuzz.token_sort_ratio, limit=limit, score_cutoff=65)
    if not results:
        return df.iloc[0:0]
    idx = [i for _, _, i in results]
    return df.loc[idx]


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="app-title">Career Timelines</div>'
        '<div class="app-subtitle">Search a name to see the career timeline built from Wikidata + Wikipedia.</div>',
        unsafe_allow_html=True,
    )

    ensure_datasets()
    datasets = list_datasets()
    if not datasets:
        st.error(f"No dataset files found in {DATA_DIR}")
        return

    with st.sidebar:
        st.subheader("Dataset")
        chosen = st.selectbox("Dataset", datasets, format_func=dataset_title, index=0)
        lang = st.radio("Language", list(LANGUAGE_LABELS), format_func=lambda l: LANGUAGE_LABELS[l], horizontal=True)
        df = select_lang(load_dataset(str(chosen)), lang)
        st.caption(f"{len(df):,} people loaded")

    query = st.text_input("Person's name", placeholder="e.g. Albert Einstein", label_visibility="collapsed")

    if not query:
        examples = ", ".join(df["display_name"].sample(min(5, len(df)), random_state=1))
        st.caption(f"Start typing a name above. A few examples: {examples}")
        return

    matches = find_matches(df, query)

    if matches.empty:
        st.warning(f"No one matching \"{query}\" found in this dataset.")
        return

    if len(matches) == 1:
        render_person(matches.iloc[0])
        return

    options = {f"{r.display_name} ({format_years(r.birthyear, r.deathyear)})": r.qid for r in matches.itertuples()}
    label = st.radio(f"{len(options)} matches — pick one:", list(options.keys()))
    selected_qid = options[label]
    render_person(df[df["qid"] == selected_qid].iloc[0])


if __name__ == "__main__":
    main()
