# File: tools/profile_pro_startup.py
"""
Profile Pro startup from a FILE PATH and optional date argument.

Examples:
  python3 tools/profile_pro_startup.py \
    --pro-path ~/Desktop/SteveHorsesPro/steve_horses_pro.py \
    --func build_cards_and_scratches \
    --date 2025-09-20 \
    --limit 40

  python3 tools/profile_pro_startup.py \
    --pro-path ~/Desktop/SteveHorsesPro/steve_horses_pro.py \
    --func build_cards \
    --date $(date +%F) \
    --limit 40
"""
from __future__ import annotations
import argparse, cProfile, importlib.util, io, pstats, time, pathlib, sys, inspect, datetime as _dt

def import_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load module from path: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # why: import by file path without PYTHONPATH
    return mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pro-path", required=True, help="Path to steve_horses_pro.py")
    ap.add_argument("--func", required=True, help="Callable inside Pro to run, e.g. build_cards_and_scratches")
    ap.add_argument("--date", default=None, help="ISO date YYYY-MM-DD; if provided and func accepts 1 arg, it will be passed")
    ap.add_argument("--limit", type=int, default=30, help="Profiler rows to print")
    args = ap.parse_args()

    pro_mod = import_from_path("pro_mod", args.pro_path)
    if not hasattr(pro_mod, args.func):
        callables = [n for n in dir(pro_mod) if callable(getattr(pro_mod, n))]
        raise SystemExit(f"Function '{args.func}' not found. Available: {', '.join(callables[:30])}")

    fn = getattr(pro_mod, args.func)

    # Decide whether to pass a date
    sig = inspect.signature(fn)
    call_args = []
    if len(sig.parameters) == 1:
        iso = args.date or _dt.date.today().isoformat()
        call_args = [iso]
        print(f"[profile] calling {args.func}({iso})")
    elif len(sig.parameters) > 1:
        raise SystemExit(f"{args.func} expects {len(sig.parameters)} params; this profiler only supports 0 or 1.")

    # Time import alone (helps spot slow module import)
    print("[profile] module imported; starting profiler...")
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    try:
        fn(*call_args)
    finally:
        pr.disable()
    dt = time.perf_counter() - t0

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(args.limit)
    print(f"\n[profile] total={dt:.3f}s, top {args.limit} by cumulative time:\n")
    print(s.getvalue())

    out = pathlib.Path("startup_profile.prof").resolve()
    pr.dump_stats(str(out))
    print(f"[profile] wrote {out}")

if __name__ == "__main__":
    main()