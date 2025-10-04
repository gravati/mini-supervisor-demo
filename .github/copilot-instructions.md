# mini-supervisor — Copilot instructions

Purpose: provide concise, actionable guidance for an AI coding agent to be productive in this repo.

Repository big picture
- This is a minimal demo app (Streamlit UI in `app.py`) that shows a "MiniSupervisor" which scores streaming events for semantic and calibration drift.
- Core supervisor logic lives in `supervisor/core.py` (class `MiniSupervisor`). It implements baseline fitting (`fit_baseline`), per-event scoring (`score_event`), storm detection, and a simple recalibration action gating.
- Small utility functions are in `supervisor/utils.py` (e.g., `emd1d`).
- Three lightweight adapter demos produce synthetic events: `supervisor/adapters/signals_demo.py`, `supervisor/adapters/text_demo.py`, and `supervisor/adapters/image_demo.py`. Each adapter implements:
  - `baseline_batch(n)` -> (vecs, aux): used to fit supervisor baseline
  - `next_event()` -> (vec, aux, meta): produces streaming events
  - `set_params(...)` to set demo knobs

What an AI agent should know first
- Tests/build: There are no automated tests or build scripts. Running locally uses `streamlit run app.py` (see `requirements.txt` for deps: numpy, streamlit, pillow, scikit-learn, pandas).
- Entry points:
  - UI/demo: `app.py` (Streamlit). To reproduce behavior quickly, run the Streamlit app and use the UI controls.
  - Programmatic: import `supervisor.core.MiniSupervisor` and adapters under `supervisor.adapters.*`.
- Data shapes and contracts (critical):
  - Baseline batch: `vecs` is a 2D array (n x D), `aux` is 1D numeric array-like.
  - Event scoring: `score_event(vec, aux)` expects `vec` shaped like baseline vectors and `aux` a scalar numeric.
  - Returned `score_event` dict contains keys: `sem_z`, `cal_z`, `fused`, `status`, `storm`, `storm_frac`, `action`.
- Side effects: `MiniSupervisor` keeps internal state (time t, storm window deque, recent fail deque). `fit_baseline` must be called before `score_event`.

Project-specific patterns and conventions
- Lightweight, readable code favored over micro-optimizations. Prefer explicit numpy usage and small helper functions (see `_robust_stats`, `_cosine`, `_z`).
- Adapters provide synthetic "realistic" signals and intentionally return non-zero aux distributions for baseline (see comments in `text_demo`/`image_demo`).
- Storm handling: When `enable_storm` is True, `score_event` will downweight calibration and optionally cap calibration z-scores. Tests/fixes that touch storming behavior should consider both storm and non-storm branches.
- Action gating: `_maybe_recalibrate` returns `REQUEST_RECAL` only after a sustained window of failures and cooldown; this stateful logic is time-indexed by `self.t`.

Common low-risk edits an AI agent can safely make
- Fix minor bugs and typos in docstrings and comments.
- Add type hints for public methods in adapters and `MiniSupervisor` (keep runtime behavior unchanged).
- Improve small utilities (e.g., add brief unit tests for `emd1d`), but do not change the numeric behavior without explicit tests.

Examples and references in repo
- To create a new adapter follow `supervisor/adapters/*`: implement `baseline_batch`, `next_event`, `set_params`.
- To run a quick smoke manual test (human):
  - Install deps: pip install -r requirements.txt
  - Run UI: streamlit run app.py

What not to change without human review
- The scoring math in `supervisor/core.py` (cosine-based semantic score, robust MAD z-scaling, fusion weights). Numeric thresholds (warn, fail, caps) are domain choices; changes can silently alter demo behavior.
- The baseline contract: `fit_baseline` must produce `centroid`, `mu_sem`, `sig_sem`, `mu_cal`, `sig_cal` expected by `score_event`.

Files to inspect when debugging
- `app.py` — UI wiring, session state handling, how baseline is fit and supervisor is used.
- `supervisor/core.py` — scoring, storm handling, action gating.
- `supervisor/adapters/*` — synthetic data generation and expected vector/aux shapes.
- `requirements.txt` — runtime dependencies for reproducing locally.

If you edit code, also update or add
- Small smoke tests or a short `tests/test_core.py` to assert baseline -> score_event runs without assertion errors (happy path).
- Update `requirements.txt` only when adding new runtime libs.

If anything is unclear in these instructions or key design goals are missing, ask the repo owner for guidance (preferred questions: intended numerical behavior changes; whether to add tests; whether to support non-synthetic adapters).
