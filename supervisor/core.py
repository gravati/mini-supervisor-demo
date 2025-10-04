from collections import deque
import numpy as np

class MiniSupervisor:
    """Minimal, readable drift supervisor.
    - Baseline fitting from quiet data
    - Per-event scoring: sem_z, cal_z, fused -> {OK, WARN, FAIL}
    - Storm handling (optional): downweight & cap calibration when sustained high drift
    - Action gating: request recalibration only on sustained FAILs + cooldown
    """
    def __init__(self,
                 w_sem=0.7, w_cal=0.3,
                 zcap_sem=3.0, zcap_cal=3.0, storm_cal_cap=1.5,
                 warn=2.0, fail=3.0,
                 storm_window=50, storm_enter=0.4, storm_exit=0.2,
                 recalc_cooldown=60, recalc_win=50, recalc_fail_frac=0.5,
                 enable_storm=True, cap_cal_in_storm=True):
        # weights & thresholds
        self.w_sem, self.w_cal = w_sem, w_cal
        self.zcap_sem, self.zcap_cal = zcap_sem, zcap_cal
        self.storm_cal_cap = storm_cal_cap
        self.warn, self.fail = warn, fail
        # storm controls
        self.enable_storm = bool(enable_storm)
        self.cap_cal_in_storm = bool(cap_cal_in_storm)
        self.storm_window, self.storm_enter, self.storm_exit = storm_window, storm_enter, storm_exit
        # action regulation
        self.recalc_cooldown, self.recalc_win, self.recalc_fail_frac = recalc_cooldown, recalc_win, recalc_fail_frac

        self._cal_high = deque(maxlen=self.storm_window)
        self._fail_recent = deque(maxlen=self.recalc_win)
        self.storming = False
        self._last_recalc_t = -1
        self.t = 0

        # learned on autocal
        self.centroid = None
        self.mu_sem = self.sig_sem = None
        self.mu_cal = self.sig_cal = None
        # semantic metric: 'cosine' or 'mahalanobis'
        self.sem_metric = 'cosine'
        # inverse covariance for Mahalanobis (learned at fit_baseline)
        self._inv_cov = None

    # --- utilities ----------------------------------------------------------
    @staticmethod
    def _robust_stats(x):
        x = np.asarray(x)
        mu = np.median(x)
        sig = 1.4826 * np.median(np.abs(x - mu)) + 1e-6
        """Floor sigma to avoid degenerate near-zero values which produce
        artificially large z-scores in small baselines.
        Uses an adaptive floor: at least 0.05 but also a fraction of the median
        so metrics with larger median distances
        don't produce huge z-scores from tiny MADs."""
        adaptive_floor = max(0.05, 0.15 * abs(float(mu)))
        sig = max(float(sig), adaptive_floor)
        return float(mu), float(sig)

    @staticmethod
    def _regularized_inv_cov(X, reg=1e-3):
        """Compute a regularized inverse covariance matrix for rows X (n x d).
        Falls back to diagonal inverse if full inverse is unstable.
        """
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            return None
        # empirical covariance (features in columns)
        cov = np.cov(X, rowvar=False)
        # scale reg relative to trace for stability across scales
        d = cov.shape[0]
        trace = np.trace(cov)
        eps = reg * (trace / max(1.0, d))
        cov_reg = cov + eps * np.eye(d)
        try:
            inv = np.linalg.inv(cov_reg)
            return inv
        except np.linalg.LinAlgError:
            # fallback: inverse of diagonal only
            diag = np.diag(cov_reg).copy()
            diag[diag <= 0] = eps
            return np.diag(1.0 / diag)

    @staticmethod
    def _cosine(a, b):
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _z(x, mu, sig, cap):
        return float(np.clip((x - mu) / sig, -cap, cap))

    # --- baseline ----------------------------------------------------------
    def fit_baseline(self, vecs, aux_vals):
        """Fit centroid and robust z-scalers from quiet baseline samples."""
        vecs = np.asarray(vecs)
        self.centroid = vecs.mean(axis=0)
        # compute semantic raw distances according to chosen metric
        if self.sem_metric == 'mahalanobis':
            # learn inverse covariance and compute mahalanobis distances
            self._inv_cov = self._regularized_inv_cov(vecs)
            if self._inv_cov is None:
                sem = [1.0 - self._cosine(v, self.centroid) for v in vecs]
            else:
                sem = [float(np.sqrt((v - self.centroid) @ (self._inv_cov @ (v - self.centroid)))) for v in vecs]
        else:
            sem = [1.0 - self._cosine(v, self.centroid) for v in vecs]
        cal = list(map(float, aux_vals))
        self.mu_sem, self.sig_sem = self._robust_stats(sem)
        self.mu_cal, self.sig_cal = self._robust_stats(cal)

    # --- storm logic -------------------------------------------------------
    def _update_storm(self, cal_z):
        if not self.enable_storm:
            # storm disabled
            self.storming = False
            self._cal_high.clear()
            return False, 0.0
        # mark high cal if cal_z exceeds 70% of the cap
        threshold = 0.7 * self.zcap_cal
        self._cal_high.append(1.0 if cal_z > threshold else 0.0)
        frac = sum(self._cal_high) / max(1, len(self._cal_high))
        if not self.storming and frac >= self.storm_enter:
            self.storming = True
        elif self.storming and frac <= self.storm_exit:
            self.storming = False
        return self.storming, frac

    # --- actions -----------------------------------------------------------
    def _maybe_recalibrate(self, status):
        if self.t - self._last_recalc_t < self.recalc_cooldown:
            return None
        if len(self._fail_recent) == self.recalc_win:
            frac = sum(self._fail_recent) / self.recalc_win
            if frac >= self.recalc_fail_frac:
                self._last_recalc_t = self.t
                return "REQUEST_RECAL"
        return None

    # --- scoring -----------------------------------------------------------
    def score_event(self, vec, aux):
        assert self.centroid is not None, "Call fit_baseline() first"
        self.t += 1
        # compute semantic raw distance according to selected metric
        if self.sem_metric == 'mahalanobis' and self._inv_cov is not None:
            d = vec - self.centroid
            sem_raw = float(np.sqrt(d @ (self._inv_cov @ d)))
        else:
            sem_raw = 1.0 - self._cosine(vec, self.centroid)
        cal_raw = float(aux)
        sem_z = self._z(sem_raw, self.mu_sem, self.sig_sem, self.zcap_sem)
        cal_z = self._z(cal_raw, self.mu_cal, self.sig_cal, self.zcap_cal)

        storm, storm_frac = self._update_storm(cal_z)
        w_sem, w_cal = (0.9, 0.1) if storm else (self.w_sem, self.w_cal)
        if storm and self.cap_cal_in_storm:
            cal_z = float(np.clip(cal_z, -self.storm_cal_cap, self.storm_cal_cap))
        fused = w_sem * sem_z + w_cal * cal_z
        status = "OK" if fused < self.warn else ("WARN" if fused < self.fail else "FAIL")
        self._fail_recent.append(1 if status == "FAIL" else 0)
        action = self._maybe_recalibrate(status)

        return {
            "sem_z": sem_z,
            "cal_z": cal_z,
            "fused": float(fused),
            "status": status,
            "storm": storm,
            "storm_frac": float(storm_frac),
            "action": action,
        }