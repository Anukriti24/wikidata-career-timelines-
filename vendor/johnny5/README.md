# johnny5 (vendored)

Vendored copy of the johnny5 occupation classifier, originally built by the Collective
Learning group at MIT Media Lab (Cristian Jara-Figueroa) for the Pantheon project.
Source: https://github.com/datawheel/johnny5, a fork/mirror of the original
https://github.com/crisjf/johnny5 - which is gone now (404 as of 2026-07-08, and was
never published on PyPI either). MIT licensed, see LICENSE in this directory.

This folder just IS the package now - no separate download or pip install of johnny5
itself, since there's nowhere left to install it from.

## Patched

One change from the upstream source: `johnny5/query.py`'s `_rget()` used to spoof an
old-browser User-Agent (`Mozilla/4.0 ... Windows 98`), which Wikidata/Wikipedia now
rate-limit. Replaced with a proper identifying User-Agent (contact email included),
matching `src/wikidata.py`/`src/wikipedia.py` elsewhere in this project. Otherwise
this is an unmodified copy of the source.

## Install

```bash
pip install nltk mwparserfromhell beautifulsoup4 python-dateutil scikit-learn
```

That's everything it needs to run.

## Run

The pretrained classifier in here (`johnny5/data/trained_classifier.pkl`) was pickled
with scikit-learn ~0.17 back in 2017, so it won't load as-is under a modern sklearn:
internal module paths got renamed (e.g. `sklearn.svm.classes` -> `sklearn.svm._classes`),
LIBSVM now needs 32-bit sparse indices instead of 64-bit, and a few fitted attributes
got renamed too (e.g. `probA_` -> `_probA`). Those fixes are applied automatically the
moment you import through `src/johnny5_baseline.py`, so use that instead of importing
`johnny5` directly:

```python
from src.johnny5_baseline import classify_qid

label, prob_ratio = classify_qid("Q937")
# -> ("PHYSICIST", 4.2)
```

`label` is johnny5's predicted broad occupation category. `prob_ratio` is how much
more confident it was in that label over its runner-up (0 if the QID was in johnny5's
own training data, in which case it just returns the known answer instead of a real
prediction).
