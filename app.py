import numpy as np
import pandas as pd
import streamlit as st

from supervisor.core import MiniSupervisor
from supervisor.adapters.signals_demo import SignalsAdapter
from supervisor.adapters.text_demo import TextAdapter
from supervisor.adapters.image_demo import ImageAdapter

st.set_page_config(page_title="mini-supervisor", layout="centered")
st.title("Mini Supervisor — Drift Demo")


# --- CONTROLS ----------------------------------------------------------
adapter_name = st.radio("Adapter", ["Signals", "Text", "Image"], horizontal=True)

# Display info textbox
if adapter_name == "Signals":
    st.info("Imagine you're a physicist operating a telescope array. Each antenna produces numeric telemetry describing its signal noise profile. Over time, temp fluctuations or hardware aging can cause those noise distributions to shift. The supervisor tracks both the feature vectors (signal shape) and the calibration histograms (sensor stability) and can warn you before the instrument degrades. Change the run type, thresholds/weights, and storm settings to see how the supervisor responds.")
elif adapter_name == "Text":
    st.info("Imagine you're monitoring a LLM 'agent' meant to operate a vending machine. The supervisor can track its 'chain of thought' and actions (such as emails to vendors) to ensure that the agent remains focused on its assigned task and isn't hallucinating or acting out of context. Change the run type, thresholds/weights, and storm settings to see how the supervisor responds.")
else:
    st.info("Imagine you're a chemist using a camera to monitor reactions through a microscope. As lighting changes or lenses fog, the representative image histograms drift. The supervisor can flag these calibration changes automatically - before your image analysis pipeline starts producing unreliable results. Change the run type, thresholds/weights, and storm settings to see how the supervisor responds.")

# Global run type to control demo modes for all adapters
run_type = st.selectbox(
    "Run type",
    ["default", "drift", "storm", "high_cal_drift", "random"],
    index=0,
    help="Preset generation modes applied to the selected adapter",
)

with st.expander("Behavior", expanded=True):
    enable_storm = st.checkbox(
        "Enable storm mode (downweight calibration when sustained high drift)", value=False
    )
    cap_cal = st.checkbox("Cap calibration z-score while storming", value=False)
    if adapter_name == "Signals":
        warn_default = 0.5
        fail_default = 1.0
        sem_default = 0.45
    elif adapter_name == "Text":
        warn_default = 0.8
        fail_default = 1.6
        sem_default = 0.45
    else:  # Image
        warn_default = 0.8
        fail_default = 1.6
        sem_default = 0.45
    # Allow users to adjust thresholds and weights
    warn = st.slider("WARN threshold (z)", 0.2, 5.0, float(warn_default), 0.1)
    fail = st.slider("FAIL threshold (z)", 0.5, 6.0, float(fail_default), 0.1)
    # Ensure warn < fail
    if warn >= fail:
        warn = max(0.5, fail - 0.1)

    sem_weight = st.slider("Semantic weight (calibration weight = 1 - this)", 0.0, 1.0, float(sem_default), 0.05)

    # Reset image baseline for drift runs
    refit_image_for_drift = True

tilt = 0.0
drift = 0.0
storm_p = 0.0

n_events = st.number_input("Events to emit", min_value=1, max_value=2000, value=200, step=10)
show_push = st.checkbox("Show push bars (fused - WARN)", value=False)

# Build / refresh supervisor + adapter when adapter changes
if "sup" not in st.session_state or st.session_state.get("adapter_name") != adapter_name:
    st.session_state.sup = MiniSupervisor(
        enable_storm=enable_storm,
        cap_cal_in_storm=cap_cal,
        warn=warn,
        fail=fail,
    )
    if adapter_name == "Signals":
        st.session_state.adapter = SignalsAdapter()
    elif adapter_name == "Text":
        st.session_state.adapter = TextAdapter()
    else:
        st.session_state.adapter = ImageAdapter()

    # apply global run type to new adapter when possible
    try:
        st.session_state.adapter.run_type = run_type
    except Exception:
        pass
    vecs, aux = st.session_state.adapter.baseline_batch(200)
    # Choose semantic metric per-adapter; Mahalanobis for Signals
    if adapter_name == "Signals":
        st.session_state.sup.sem_metric = 'mahalanobis'
    else:
        st.session_state.sup.sem_metric = 'cosine'
    st.session_state.sup.fit_baseline(vecs, aux)
    st.session_state.adapter_name = adapter_name

# Keep behavior knobs in sync on every rerun
sup = st.session_state.sup
sup.enable_storm = enable_storm
sup.cap_cal_in_storm = cap_cal
sup.warn = warn
sup.fail = fail
sup.w_sem = float(sem_weight)
sup.w_cal = float(1.0 - sem_weight)

ad = st.session_state.adapter
try:
    ad.run_type = run_type
except Exception:
    pass
# Forward sensible defaults for the chosen run_type so effects are visible
try:
    if run_type == 'default':
        # minimal/noise settings
        if adapter_name == 'Signals':
            ad.set_params(topic_drift=0.0, cal_tilt=0.0, storm_p=0.0)
        elif adapter_name == 'Text':
            ad.set_params(topic_shift=0.0, noise=0.0, storm_p=0.0)
        else:  # Image
            ad.set_params(contrast=1.0, bright=0.0, storm_p=0.0)
    if run_type == 'drift':
        # stronger defaults so ramp is visually apparent
        if adapter_name == 'Signals':
            ad.set_params(topic_drift=2.0, cal_tilt=0.3, storm_p=0.0)
        elif adapter_name == 'Text':
            ad.set_params(topic_shift=1.5, noise=0.35, storm_p=0.0)
        else:  # Image
            ad.set_params(contrast=1.25, bright=0.05, storm_p=0.0)
    elif run_type == 'high_cal_drift':
        if adapter_name == 'Signals':
            ad.set_params(topic_drift=0.1, cal_tilt=1.6, storm_p=0.0)
        elif adapter_name == 'Text':
            ad.set_params(topic_shift=0.05, noise=0.02, storm_p=0.0)
        else:
            ad.set_params(contrast=1.0, bright=0.35, storm_p=0.0)
    elif run_type == 'high_semantic_drift':
        if adapter_name == 'Signals':
            ad.set_params(topic_drift=1.8, cal_tilt=0.05, storm_p=0.0)
        elif adapter_name == 'Text':
            ad.set_params(topic_shift=1.0, noise=0.2, storm_p=0.0)
        else:
            ad.set_params(contrast=1.8, bright=0.0, storm_p=0.0)
    elif run_type == 'random':
        # leave adapters to sample their own random params; enable small storm chance
        ad.set_params(storm_p=0.05)
except Exception:
    pass

# Run
force_reset = st.checkbox("Force reset baseline", value=False)

if st.button("Run"):
    rows = []
    last_out = None
    # Only refit baseline if user explicitly requested a force reset
    if force_reset:
        try:
            vecs, aux = ad.baseline_batch(200, force_recompute=True)
            # Ensure semantic metric matches the current adapter before fitting
            sup.sem_metric = 'mahalanobis' if adapter_name == "Signals" else 'cosine'
            sup.fit_baseline(vecs, aux)
            st.info("Baseline refit on 200 samples before run.")
        except Exception:
            pass
        
    try:
        ad._run_start = getattr(ad, "_t", 0)
    except Exception:
        pass
    # For image-like adapters, refit the baseline using the current adapter params
    try:
        if adapter_name == 'Image' and run_type in ('drift', 'high_cal_drift', 'high_semantic_drift', 'random'):
            vecs, aux = ad.baseline_batch(200, force_recompute=True)
            sup.sem_metric = 'cosine'
            sup.fit_baseline(vecs, aux)
    except Exception:
        pass
    for _ in range(int(n_events)):
        v, a, meta = ad.next_event()
        out = sup.score_event(v, a)
        last_out = out
        # capture adapter-provided metadata when available
        t = None
        storm_active = False
        try:
            if isinstance(meta, dict):
                t = meta.get("t", None)
                storm_active = bool(meta.get("storm_active", False))
        except Exception:
            pass
        rows.append({
            "sem_z": float(out["sem_z"]),
            "cal_z": float(out["cal_z"]),
            "fused": float(out["fused"]),
            "status": str(out["status"]),
            "t": t,
            "storm_active": storm_active,
        })


    df = pd.DataFrame(rows)
    st.session_state["results_df"] = df

    # Timeline helpers: rolling mean to reveal drift-over-time
    try:
        df["fused_roll"] = df["fused"].rolling(window=10, min_periods=1).mean()
    except Exception:
        df["fused_roll"] = df["fused"]

    st.subheader("Results")
    st.metric(
        "Storming",
        "Yes" if last_out and last_out["storm"] else "No",
        help=(f"Recent high cal frac: {float(last_out['storm_frac']):.2f}" if last_out else None),
    )

    # Visual: per-event push past WARN + fused points colored by status
    df = df.reset_index(drop=True)
    df["idx"] = df.index.astype(int)
    df["push"] = df["fused"] - float(warn)

    try:
        layers = []

        if show_push:
            bar_layer = {
                "transform": [{"filter": "datum.push > 0"}],
                "mark": {"type": "bar", "opacity": 0.5},
                "encoding": {
                    "x": {"field": "idx", "type": "quantitative", "axis": {"labelAngle": 0, "tickCount": 10}},
                    "y": {"field": "push", "type": "quantitative", "title": "push (fused - WARN)"},
                    "color": {"field": "status", "type": "nominal",
                              "scale": {"domain": ["OK", "WARN", "FAIL"],
                                        "range": ["#a6cee3", "#ffb74d", "#e57373"]},
                              "legend": None}
                }
            }
            layers.append(bar_layer)

        # storm shading
        shade_layer = {
            "transform": [{"filter": "datum.storm_active == true"}],
            "mark": {"type": "rect", "opacity": 0.12, "color": "#6a51a3"},
            "encoding": {
                "x": {"field": "idx", "type": "quantitative"},
                "x2": {"field": "idx", "type": "quantitative"},
            },
        }
        
        layers.insert(0, shade_layer)

        point_layer = {
            "mark": {"type": "point", "filled": True, "size": 110, "stroke": "#222", "strokeWidth": 0.8},
            "encoding": {
                "x": {"field": "idx", "type": "quantitative", "title": "Event", "axis": {"labelAngle": 0, "tickCount": 10}},
                "y": {"field": "fused", "type": "quantitative", "title": "fused"},
                "color": {"field": "status", "type": "nominal",
                          "scale": {"domain": ["OK", "WARN", "FAIL"],
                                    "range": ["#1f77b4", "#ff7f0e", "#d62728"]},
                          "legend": {"title": "status", "orient": "right"}},
                "tooltip": [
                    {"field": "idx"}, {"field": "sem_z"}, {"field": "cal_z"},
                    {"field": "fused"}, {"field": "status"}, {"field": "push"}
                ]
            }
        }
        layers.append(point_layer)

        layers.append({"mark": {"type": "rule", "color": "#0b84a5", "strokeWidth": 2, "opacity": 0.95},
                       "encoding": {"y": {"datum": float(warn)}}})
        layers.append({"mark": {"type": "rule", "color": "#d11149", "strokeWidth": 2, "opacity": 0.95},
                       "encoding": {"y": {"datum": float(fail)}}})

        spec = {
            "width": "container",
            "height": 380,
            "layer": layers,
            "resolve": {"scale": {"y": ("independent" if show_push else "shared")} },
            "config": {"axis": {"labelFontSize": 11, "titleFontSize": 12}}
        }

        st.vega_lite_chart(df.astype(object), spec, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render per-event chart: {e}")
        st.bar_chart(df[["sem_z", "cal_z"]], height=200)
        st.line_chart(df["fused"].rename("fused"), height=200)

    # Timeline: fused over time with rolling mean
    try:
        ts_spec = {
            "width": "container",
            "height": 180,
            "layer": [
                {   # raw fused
                    "mark": {"type": "line", "opacity": 0.45},
                    "encoding": {
                        "x": {"field": "idx", "type": "quantitative", "title": "Event"},
                        "y": {"field": "fused", "type": "quantitative", "title": "fused"}
                    }
                },
                {   # rolling mean
                    "mark": {"type": "line", "color": "#f59e0b", "strokeWidth": 3},
                    "encoding": {
                        "x": {"field": "idx", "type": "quantitative"},
                        "y": {"field": "fused_roll", "type": "quantitative"}
                    }
                },
                {   # points colored by status 
                    "mark": {"type": "point", "filled": True, "size": 40},
                    "encoding": {
                        "x": {"field": "idx", "type": "quantitative"},
                        "y": {"field": "fused", "type": "quantitative"},
                        "color": {"field": "status", "type": "nominal",
                                  "scale": {"domain": ["OK", "WARN", "FAIL"],
                                            "range": ["#1f77b4", "#ff7f0e", "#d62728"]},
                                  "legend": {"title": "status", "orient": "right"}}
                    }
                },
                {"mark": {"type": "rule", "color": "#0b84a5", "strokeWidth": 1, "opacity": 0.9},
                 "encoding": {"y": {"datum": float(warn)}}},
                {"mark": {"type": "rule", "color": "#d11149", "strokeWidth": 1, "opacity": 0.9},
                 "encoding": {"y": {"datum": float(fail)}}}
            ]
        }
        st.vega_lite_chart(df.astype(object), ts_spec, use_container_width=True)
    except Exception:
        # best-effort fallback
        st.line_chart(df["fused"], height=180)

    # Counts from the status column
    counts = df["status"].value_counts().reindex(["OK", "WARN", "FAIL"], fill_value=0).astype(int).to_dict()
    st.session_state["last_counts"] = counts
    st.json(counts)

with st.expander("How is synthetic data generated?", expanded = False):
    if adapter_name == 'Signals':
        st.info("The signals adapter generates random numeric vectors centered around a baseline centroid. The centroid shifts over time (semantic drift), and a separate histogram of noise amplitudes (aux) shifts or spikes (calibration drift). Short bursts simulate 'storms' where calibration channels are unstable.")
    elif adapter_name == 'Text':
        st.info("The text adapter uses tiny TF-IDF vectors from a glossary of technical terms. It will inject incorrect puncutation, padding, or off-topic phrases into events depending on run type. These manipulations simulate the text shifting topics (semantic drift) and changing structure (calibration drift via text length).")
    else:
        st.info("The image adapter produces small grayscale noise images and computes 32-bin intensity histograms to represent features. Adjusting brightness and contrast simulates optical degradation or exposure errors. Randomized bright patches simulate short bursts of sensor saturation.")



st.write("Interested in implementing this architecture in your data pipeline? Let's talk! **hlyons2@luc.edu**")