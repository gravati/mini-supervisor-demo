import numpy as np
from supervisor.utils import emd1d


class SignalsAdapter:
    """Synthetic numeric stream producing (vec, aux, meta).

    - vec: d-dimensional gaussian around a centroid
    - aux: calibration proxy = emd between current noise histogram and a baseline
    """

    def __init__(self, dim: int = 32, seed: int = 1234, bins: int = 32):
        self.dim = dim
        self.bins = bins
        self.rng = np.random.default_rng(seed)

        # baseline centroid
        self.base_centroid = self.rng.normal(0.0, 1.0, size=dim)

        # baseline histogram for calibration (reference noise)
        self.base_hist = np.histogram(
            self.rng.normal(0, 1, size=2048),
            bins=self.bins,
            range=(-4, 4),
        )[0]

        # user-tunable knobs (opt.)
        self._topic_drift = 0.0  # semantic drift strength
        self._cal_tilt = 0.0     # mean shift for aux distribution
        self._storm_p = 0.0      # per-event aux burst probability
        # run_type: 'default', 'drift', 'storm', 'high_cal_drift', 'high_semantic_drift', 'random'
        self.run_type = 'default'
        self._rand_params = None
        self._t = 0
        self._storm_active = False
        self._storm_remaining = 0
        # baseline cache keyed by n -> (vecs, aux)
        self._baseline_cache = {}

    # public controls
    def set_params(self, topic_drift=None, cal_tilt=None, storm_p=None):
        if topic_drift is not None:
            self._topic_drift = float(topic_drift)
        if cal_tilt is not None:
            self._cal_tilt = float(cal_tilt)
        if storm_p is not None:
            self._storm_p = float(storm_p)
        # if caller passes a string as topic_drift treat it as run_type (backwards compat)
        if isinstance(topic_drift, str):
            self.run_type = topic_drift
            # reset random params when switching to random mode
            if self.run_type != 'random':
                self._rand_params = None

    # baseline batch: emit quiet samples for fitting
    def baseline_batch(self, n: int = 500, force_recompute: bool = False):
        key = int(n)
        if (not force_recompute) and key in self._baseline_cache:
            return self._baseline_cache[key]

        vecs = []
        aux = []
        for _ in range(n):
            # Semantic vectors ~ N(mu=base_centroid, I)
            v = self.rng.normal(0, 1, size=self.dim) + self.base_centroid
            vecs.append(v)

            # Aux histograms from N(0,1); compute EMD to baseline
            h = np.histogram(
                self.rng.normal(0, 1, size=1024),
                bins=self.bins,
                range=(-4, 4),
            )[0]
            aux.append(emd1d(h, self.base_hist))

        out = (np.asarray(vecs, dtype=float), np.asarray(aux, dtype=float))
        self._baseline_cache[key] = out
        return out

    # streaming event generator
    def next_event(self):
        """Return (vec, aux, meta) for one synthetic event.

        vec: semantic vector (centroid + orthogonal drift)
        aux: calibration proxy (EMD to baseline histogram)
        meta: dict of parameters/state
        """
        mean_shift = 0.0
        mu = 0.0
        
        if self.run_type == 'default':
            mean_shift = self._topic_drift * 1.5
            mu = self._cal_tilt * 1.0
            # occasional small burst
            if self.rng.random() < 0.01:
                # small aux burst
                mu += 1.5


        elif self.run_type == 'drift':
            run_start = getattr(self, "_run_start", 0)
            rel_t = max(0, int(self._t) - int(run_start))
            frac = min(1.0, float(rel_t) / 200.0)
            mean_shift = self._topic_drift * 4.0 * frac
            mu = self._cal_tilt * 2.0 * frac


        elif self.run_type == 'storm':
            # start a storm window occasionally
            if not self._storm_active and self.rng.random() < 0.06:
                self._storm_active = True
                self._storm_remaining = int(self.rng.integers(6, 36))
            if self._storm_active:
                # heavy semantic drift and calibration shift during storms
                mean_shift = self._topic_drift * 4.0 + 2.0
                mu = self._cal_tilt * 2.0 + 2.5
                self._storm_remaining -= 1
                if self._storm_remaining <= 0:
                    self._storm_active = False
            else:
                mean_shift = self._topic_drift * 1.0
                mu = self._cal_tilt * 1.0


        elif self.run_type == 'high_cal_drift':
            mean_shift = self._topic_drift * 0.5
            mu = 2.0 + abs(self._cal_tilt) * 2.0


        elif self.run_type == 'high_semantic_drift':
            mean_shift = 3.0 + abs(self._topic_drift) * 3.0
            mu = self._cal_tilt * 0.4


        elif self.run_type == 'random':
            if self._rand_params is None:
                # sample persistent random scales for this run
                self._rand_params = {
                    'topic_scale': float(self.rng.uniform(0.2, 4.0)),
                    'cal_scale': float(self.rng.uniform(0.2, 4.0)),
                    'storm_scale': float(self.rng.random()),
                }
            mean_shift = self._rand_params['topic_scale']
            mu = self._rand_params['cal_scale']

    # unit vector along centroid
        u = self.base_centroid / (np.linalg.norm(self.base_centroid) + 1e-9)

    # random direction orthogonal to centroid
        r = self.rng.normal(0, 1, size=self.dim)
        r = r - (r @ u) * u
        r = r / (np.linalg.norm(r) + 1e-9)

    # final vector = noise + centroid + orthogonal drift
        vec = self.rng.normal(0, 1, size=self.dim) + self.base_centroid + mean_shift * r

        # calibration aux: histogram centered at mu; optional burst
        size = 1024
        x = self.rng.normal(mu, 1, size=size)
        if self.rng.random() < self._storm_p:
            x[: size // 4] = self.rng.normal(mu + 2.5, 0.5, size=size // 4)
        h = np.histogram(x, bins=self.bins, range=(-4, 4))[0]
        aux = emd1d(h, self.base_hist)

        meta = {
            "topic_drift": self._topic_drift,
            "cal_tilt": self._cal_tilt,
            "storm": self._storm_p,
            "run_type": self.run_type,
            "t": int(self._t),
            "storm_active": bool(self._storm_active),
        }

        # advance time
        self._t += 1

        return vec.astype(float), float(aux), meta
