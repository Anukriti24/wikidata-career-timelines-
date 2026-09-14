import sys
import signal
import importlib
import warnings
from pathlib import Path
from functools import lru_cache

VENDOR_PATH = Path(__file__).resolve().parent.parent / "vendor" / "johnny5"
if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))

# --- Shim 1: old sklearn internal module paths were renamed to private (underscore) ones ---
_MODULE_ALIASES = [
    ("sklearn.svm.classes", "sklearn.svm._classes"),
    ("sklearn.preprocessing.label", "sklearn.preprocessing._label"),
    ("sklearn.feature_extraction.dict_vectorizer", "sklearn.feature_extraction._dict_vectorizer"),
]
for _old, _new in _MODULE_ALIASES:
    sys.modules.setdefault(_old, importlib.import_module(_new))

warnings.filterwarnings("ignore")

import nltk.classify.scikitlearn as _sk_mod  # noqa: E402

def _heal_attr(clf, missing_name: str) -> bool:
    """Bridge old public fitted-attribute names (e.g. 'probA_') to their new
    private equivalents (e.g. '_probA'), which modern sklearn's SVC expects."""
    if missing_name.startswith("_"):
        candidate = missing_name[1:] + "_"
        if candidate in clf.__dict__:
            clf.__dict__[missing_name] = clf.__dict__[candidate]
            return True
    return False

def prob_classify_many(self, featuresets):
    # Shim 2: LIBSVM (used internally by the pickled SVC) requires 32-bit sparse
    # indices; modern scipy/DictVectorizer default to 64-bit.
    X = self._vectorizer.transform(featuresets)
    if hasattr(X, "indices"):
        X.indices = X.indices.astype("int32")
        X.indptr = X.indptr.astype("int32")

    clf = self._clf
    # Shim 3: sklearn 1.9 introduced this attribute at fit-time; our object was
    # never fit in this process, so it's simply missing.
    clf.__dict__.setdefault("_effective_probability", True)

    for _ in range(20):
        try:
            y_proba_list = clf.predict_proba(X)
            break
        except AttributeError as e:
            if not _heal_attr(clf, e.name):
                raise
    else:
        raise RuntimeError("exhausted attribute-healing attempts while unpickling classifier")

    return [self._make_probdist(yp) for yp in y_proba_list]

_sk_mod.SklearnClassifier.prob_classify_many = prob_classify_many

import johnny5 as j5  # noqa: E402

def _decode(label):
    return label.decode() if isinstance(label, bytes) else label

@lru_cache(maxsize=1)
def _classifier() -> "j5.Occ":
    return j5.Occ()

def _alarm_handler(signum, frame):
    raise TimeoutError("classify_qid timed out")

def classify_qid(qid: str, timeout: int = 30) -> tuple[str, float]:
    """
    Classify a person's primary occupation from their Wikidata QID using the
    Johnny5 baseline classifier.

    Returns
    -------
    label : str
        Predicted Pantheon-style broad occupation category (e.g. "PHYSICIST").
    prob_ratio : float
        Ratio between the top two predicted class probabilities. 0 if the QID
        was in Johnny5's own training set (see johnny5.classes.Occ.classify).

    Raises
    ------
    TimeoutError
        If the underlying Biography fetch (network I/O, no timeout of its own)
        doesn't complete within `timeout` seconds, instead of hanging forever.
    """
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        bio = j5.Biography(qid)
        # override_train=True: without it, QIDs in Johnny5's own training set (built
        # from Pantheon labels) short-circuit to the Pantheon ground truth instead of
        # an actual model prediction, which would make this baseline circular.
        label, prob_ratio = _classifier().classify(bio, override_train=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return _decode(label), prob_ratio
