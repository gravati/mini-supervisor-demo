import json
from supervisor.core import MiniSupervisor
from supervisor.adapters.signals_demo import SignalsAdapter

HELP = """
Commands:
  help                     Show help
  autocal [N]              Fit baseline on N quiet samples (default 500)
  drift X                  Set semantic/topic drift in [0..1]
  tilt X                   Set calibration tilt in [-0.5..0.5]
  storm P                  Set storm probability per event in [0..1]
  emit N                   Generate & score N events
  show                     Print current params
  quit/exit                Leave
"""

def main():
    adapter = SignalsAdapter()
    sup = MiniSupervisor()
    fitted = False
    print("mini-supervisor CLI — type 'help' to begin")

    while True:
        try:
            cmd = input("[ mini ]> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            print() ; break
        if not cmd: continue
        c = cmd[0].lower()
        if c == "help":
            print(HELP)
        elif c == "autocal":
            n = int(cmd[1]) if len(cmd) > 1 else 500
            vecs, aux = adapter.baseline_batch(n)
            sup.fit_baseline(vecs, aux)
            fitted = True
            print(f"baseline fit on {n} samples")
        elif c == "drift":
            adapter.set_params(topic_drift=float(cmd[1]))
        elif c == "tilt":
            adapter.set_params(cal_tilt=float(cmd[1]))
        elif c == "storm":
            adapter.set_params(storm_p=float(cmd[1]))
        elif c == "emit":
            if not fitted:
                print("run 'autocal' first")
                continue
            n = int(cmd[1]) if len(cmd) > 1 else 50
            for _ in range(n):
                vec, aux, meta = adapter.next_event()
                out = sup.score_event(vec, aux)
                print(json.dumps({**meta, **out}))
        elif c in ("show",):
            print(json.dumps({"topic_drift": adapter._topic_drift,
                               "cal_tilt": adapter._cal_tilt,
                               "storm_p": adapter._storm_p}, indent=2))
        elif c in ("quit", "exit"):
            break
        else:
            print("unknown command; type 'help'")

if __name__ == "__main__":
    main()