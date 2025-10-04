from typing import Tuple
import numpy as np
from PIL import Image

class ImageAdapter:
    """Image adapter using tiny grayscale histogram features.
    - baseline: random noise images (or user-uploaded set in a real demo)
    - vec: normalized 32-bin grayscale histogram
    - aux: brightness drift (abs(mean - mean_baseline))
    """
    def __init__(self, dim_bins: int = 32, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.bins = dim_bins
        self._mean_mu = 0.5
        self._mean_sig = 0.1
        self._contrast = 1.0
        self._bright = 0.0
        self._storm_p = 0.0
        # run_type and time state
        self.run_type = 'default'
        self._rand_params = None
        self._t = 0
        self._storm_active = False
        self._storm_remaining = 0
        self._base_hist = None
        self._baseline_cache = {}

    def set_params(self, contrast=None, bright=None, storm_p=None):
        if contrast is not None: self._contrast = float(contrast)
        if bright is not None: self._bright = float(bright)
        if storm_p is not None: self._storm_p = float(storm_p)
        # allow passing run_type via contrast (backwards compat if needed)
        if isinstance(contrast, str):
            self.run_type = contrast

    def _robust(self, xs):
        x = np.asarray(xs, dtype=float)
        mu = float(np.median(x))
        sig = float(1.4826 * np.median(np.abs(x - mu)) + 1e-6)
        return mu, sig

    def _hist(self, img):
        h, _ = np.histogram(img, bins=self.bins, range=(0, 1))
        h = h.astype(float) / (h.sum() + 1e-9)
        return h

    def baseline_batch(self, n: int = 20, force_recompute: bool = False):
        key = int(n)
        if (not force_recompute) and key in self._baseline_cache:
            return self._baseline_cache[key]
        imgs = self.rng.random((n, 64, 64))
        # apply current contrast/brightness settings to baseline images so
        # the baseline reflects the adapter's starting visual state
        contrast = self._contrast
        bright = self._bright
        imgs = np.clip((imgs - 0.5) * contrast + 0.5 + bright, 0, 1)
        hists = [self._hist(im) for im in imgs]
        self._base_hist = np.mean(hists, axis=0)
        means = [im.mean() for im in imgs]
        self._mean_mu, self._mean_sig = self._robust(means)
        vecs = np.array(hists, dtype=float)
        aux = np.array([abs(m - self._mean_mu) / self._mean_sig for m in means], dtype=float)
        out = (vecs, aux)
        self._baseline_cache[key] = out
        return out

    def _synth(self, size: Tuple[int, int] = (64, 64)):
        im = self.rng.random(size)
        # adjust contrast/brightness by run_type, time
        contrast = self._contrast
        bright = self._bright
        if self.run_type == 'default':
            # tiny occasional bright pixels
            if self.rng.random() < 0.03:
                bright += 0.05
        elif self.run_type == 'drift':
            run_start = getattr(self, "_run_start", 0)
            rel_t = max(0, int(self._t) - int(run_start))
            frac = min(1.0, float(rel_t) / 200.0)
            contrast = contrast * (1.0 + 0.5 * frac)
            bright = bright + 0.2 * frac
        elif self.run_type == 'storm':
            if not self._storm_active and self.rng.random() < 0.05:
                self._storm_active = True
                self._storm_remaining = int(self.rng.integers(6, 36))
            if self._storm_active:
                # strong brightening
                bright += 0.4
                contrast *= 1.5
                self._storm_remaining -= 1
                if self._storm_remaining <= 0:
                    self._storm_active = False

        elif self.run_type == 'high_cal_drift':
            # brighten images persistently to drive cal aux
            bright += 0.3

        elif self.run_type == 'high_semantic_drift':
            # increase contrast strongly to change hist shape
            contrast *= 1.8

        elif self.run_type == 'random':
            if self._rand_params is None:
                self._rand_params = {
                    'contrast': float(self.rng.uniform(0.6, 2.0)),
                    'bright': float(self.rng.uniform(-0.2, 0.6)),
                    'burst_p': float(self.rng.random()),
                }
            contrast = self._rand_params['contrast']
            bright += self._rand_params['bright']

        im = np.clip((im - 0.5) * contrast + 0.5 + bright, 0, 1)
        # per-event random short storms
        if self.rng.random() < self._storm_p:
            x0 = self.rng.integers(0, size[0] // 2)
            y0 = self.rng.integers(0, size[1] // 2)
            im[x0 : x0 + 16, y0 : y0 + 16] = 1.0
        # advance time
        self._t += 1
        return im

    def next_event(self):
        im = self._synth()
        hist = self._hist(im)
        vec = hist
        aux = abs(im.mean() - self._mean_mu) / self._mean_sig
        meta = {"contrast": self._contrast, "bright": self._bright, "storm": self._storm_p, "run_type": self.run_type, "t": int(self._t), "storm_active": bool(self._storm_active)}
        return vec, float(aux), meta