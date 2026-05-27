#!/usr/bin/env python3
"""Build a fully static snapshot of the bit-dqn dashboard for GitHub Pages.

This script reuses the data-loading logic in ``../board/server.py`` (so the
single source of truth stays there) and writes the JSON payloads that the
GitHub-Pages version of ``dashboard.html`` fetches as plain static files:

    board_github/
        api/
            runs.json
            examples/
                <variant>.json   # only for variants that have examples.json

Usage::

    cd /mnt/shenzhen2cephfs/mm-base-vision/suzzetehu/phd27fall
    python3 board_github/build_static.py

    # Optional overrides (same names as server.py):
    #   RESULTS_DIR=/path/to/results/bit-dqn python3 board_github/build_static.py

After running, commit ``board_github/`` (including ``api/``) and enable
GitHub Pages on the repo to serve ``board_github/dashboard.html``.
"""
from __future__ import annotations

import json
import os
import sys

THIS_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
BOARD_DIR = os.path.join(PROJECT_ROOT, "board")

# Import the original board/server.py without modifying it.
sys.path.insert(0, BOARD_DIR)
import server as board_server  # type: ignore  # noqa: E402


OUT_DIR = os.path.join(THIS_DIR, "api")
EXAMPLES_DIR = os.path.join(OUT_DIR, "examples")


def _write_json(path: str, payload) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return len(body)


def main() -> None:
    print(f"Results dir : {board_server.RESULTS_DIR}")
    print(f"Output dir  : {OUT_DIR}")
    if not os.path.isdir(board_server.RESULTS_DIR):
        raise SystemExit(f"RESULTS_DIR not found: {board_server.RESULTS_DIR}")

    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    runs_payload = board_server._api_runs()
    # The original API returns the absolute path under ``results_dir``; for a
    # public/static snapshot that is misleading (the path lives on the dev
    # machine), so we replace it with a stable, repo-relative hint.
    runs_payload["results_dir"] = "results/bit-dqn (static snapshot)"

    runs_path = os.path.join(OUT_DIR, "runs.json")
    size = _write_json(runs_path, runs_payload)
    print(f"  wrote {os.path.relpath(runs_path, THIS_DIR):40s}  {size/1024:8.1f} KiB")

    n_examples = 0
    total_bytes = 0
    for variant in runs_payload.get("variants", []):
        if not variant.get("has_examples"):
            continue
        name = variant["name"]
        payload = board_server._api_examples(name)
        if "error" in payload:
            print(f"  skip  {name}: {payload['error']}")
            continue
        out_path = os.path.join(EXAMPLES_DIR, f"{name}.json")
        size = _write_json(out_path, payload)
        n_examples += 1
        total_bytes += size
        print(f"  wrote {os.path.relpath(out_path, THIS_DIR):40s}  {size/1024:8.1f} KiB")

    print(f"\nDone. {n_examples} examples files, total {total_bytes/1024:.1f} KiB.")
    print(f"Open locally: python3 -m http.server -d {THIS_DIR} 8000")
    print("         then http://localhost:8000/dashboard.html")


if __name__ == "__main__":
    main()
