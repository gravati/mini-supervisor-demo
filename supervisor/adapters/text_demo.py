from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TextAdapter:
    """Tiny text adapter using TF-IDF.
    - baseline: user's glossary sentences
    - vec: TF-IDF vector of text
    - aux: format drift proxy = |len(text)-len_mean| / len_std (robustified)
    """
    def __init__(self, seed: int = 1234, baseline_perturb_p: float = 0.3):
        self.rng = np.random.default_rng(seed)
        self.vectorizer = None
        self._len_mu = 0.0
        self._len_sig = 1.0
        self._topic_shift = 0.0
        self._noise_level = 0.0
        self._storm_p = 0.0
        # run_type controls event generation behaviour: 'default', 'drift', 'storm'
        self.run_type = 'default'
        self._rand_params = None
        # internal event counter for time-varying runs
        self._t = 0
        # storm state for 'storm' mode
        self._storm_active = False
        self._storm_remaining = 0
        # small probability of variance
        self._baseline_perturb_p = float(baseline_perturb_p)
        # cache for baseline batches
        self._baseline_cache = {}

    def set_params(self, topic_shift=None, noise=None, storm_p=None, baseline_perturb_p=None):
        if topic_shift is not None: self._topic_shift = float(topic_shift)
        if noise is not None: self._noise_level = float(noise)
        if storm_p is not None: self._storm_p = float(storm_p)
        if baseline_perturb_p is not None:
            self._baseline_perturb_p = float(baseline_perturb_p)
        # accept run_type string to switch generation modes
        if isinstance(topic_shift, str):
            self.run_type = topic_shift
            if self.run_type != 'random':
                self._rand_params = None

    def _robust(self, xs: List[int]) -> Tuple[float, float]:
        x = np.asarray(xs, dtype=float)
        mu = float(np.median(x))
        sig = float(1.4826 * np.median(np.abs(x - mu)) + 1e-6)
        return mu, sig

    def baseline_batch(self, n: int = 20, glossary: List[str] | None = None, force_recompute: bool = False):
        if not glossary:
            glossary = [
                "radio telescope baseline noise",
                "clean bandpass reference data",
                "signal feature extraction and centroids",
                "semantic drift detection using cosine",
                "calibration drift measured by histogram distance",
            ]
        # fit vectorizer on glossary
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        self.vectorizer.fit(glossary)
        # create n baseline texts by cycling through glossary (quiet samples)
        key = (int(n), float(self._baseline_perturb_p))
        if (not force_recompute) and key in self._baseline_cache:
            return self._baseline_cache[key]

        texts = []
        perturb_words = ["baseline", "noise", "reference", "sample", "token"]
        for i in range(n):
            base = glossary[i % len(glossary)]
            s = base
            # small perturbation with low probability: insert a short word
            if self.rng.random() < self._baseline_perturb_p:
                w = self.rng.choice(perturb_words)
                j = self.rng.integers(0, len(s) + 1)
                s = s[:j] + " " + w + " " + s[j:]
            texts.append(s)
        X = self.vectorizer.transform(texts)
        vecs = X.toarray().astype(float)
        lengths = [len(s) for s in texts]
        self._len_mu, self._len_sig = self._robust(lengths)
        aux = np.array([abs(len(s) - self._len_mu) / self._len_sig for s in texts], dtype=float)
        self._baseline_cache[key] = (vecs, aux)
        return vecs, aux

    def _make_text(self, base: str) -> str:
        # run-type driven generation. Three modes:
        # - 'default': mostly quiet with occasional small bursts
        # - 'drift': noise increases over time (self._t)
        # - 'storm': stochastic long storms that last multiple events
        text = base
        if self.run_type == 'default':
            # small occasional punctuation noise
            if self.rng.random() < 0.8:
                i = self.rng.integers(0, len(text) + 1)
                text = text[:i] + "," + text[i:]
            # rare short bursts
            if self.rng.random() < 0.02:
                text += " " + ("x" * 30)

        elif self.run_type == 'drift':
            # ramp noise up over time using a run-relative counter
            run_start = getattr(self, "_run_start", 0)
            rel_t = max(0, int(self._t) - int(run_start))
            frac = min(1.0, float(rel_t) / 200.0)
            # topic drift grows with frac
            k = int(3 * frac)
            if k > 0:
                text += " " + " ".join(["financial markets drift" for _ in range(k)])
            # insert punctuation proportional to frac
            k2 = int(10 * frac)
            for _ in range(k2):
                i = self.rng.integers(0, len(text) + 1)
                text = text[:i] + "," + text[i:]

        elif self.run_type == 'storm':
            # possibly start a storm window
            if not self._storm_active and self.rng.random() < 0.06:
                # storm lasting a random number of events
                self._storm_active = True
                self._storm_remaining = int(self.rng.integers(8, 36))
            if self._storm_active:
                # heavy corruption during storm
                text += " " + ("x" * 200)
                # extra punctuation
                for _ in range(20):
                    i = self.rng.integers(0, len(text) + 1)
                    text = text[:i] + "," + text[i:]
                # consume storm counter
                self._storm_remaining -= 1
                if self._storm_remaining <= 0:
                    self._storm_active = False

        elif self.run_type == 'high_cal_drift':
            # consistently inject long padding that increases length-based aux
            text += " " + ("y" * 120)

        elif self.run_type == 'high_semantic_drift':
            # repeatedly append topical phrases to shift TF-IDF strongly
            text += " " + " ".join(["economic growth" for _ in range(6)])

        elif self.run_type == 'random':
            if self._rand_params is None:
                # choose a random behavior mix
                self._rand_params = {
                    'add_punct': float(self.rng.random()),
                    'append_words': int(self.rng.integers(0, 8)),
                    'burst': float(self.rng.random()),
                }
            rp = self._rand_params
            if self.rng.random() < rp['add_punct']:
                i = self.rng.integers(0, len(text) + 1)
                text = text[:i] + ";" + text[i:]
            if rp['append_words'] > 0:
                text += " " + " ".join(["market" for _ in range(rp['append_words'])])
            if self.rng.random() < rp['burst'] * 0.25:
                text += " " + ("x" * int(50 + 150 * self.rng.random()))

        # classical per-event random storm flag
        if self.rng.random() < self._storm_p:
            text += " " + ("x" * 200)

        return text

    def next_event(self):
        base = "semantic centroid comparison for baseline truth"
        s = self._make_text(base)
        # increment event counter for time-varying modes
        self._t += 1
        vec = self.vectorizer.transform([s]).toarray()[0]
        aux = abs(len(s) - self._len_mu) / self._len_sig
        meta = {"run_type": self.run_type, "t": int(self._t), "storm_active": bool(self._storm_active)}
        return vec, float(aux), meta