#!/usr/bin/env python3
"""Bit-DQN experiment dashboard server.

Scans ``results/bit-dqn/<variant>/`` for completed runs and exposes a small
read-only HTTP API consumed by ``dashboard.html``. Inspired by
``Flow-Factory/ablation/sde_windows/board/server.py`` but tailored to the
phd27fall bit-sequence experiments.

Usage:
    cd /mnt/shenzhen2cephfs/mm-base-vision/suzzetehu/phd27fall/board
    export PORT=1942
    python3 server.py

    # Optional overrides:
    #   RESULTS_DIR  -- variants root (defaults to ../results/bit-dqn)

Then open in your browser::

    http://21.6.253.14.devcloud.woa.com:${PORT}/dashboard.html
"""
from __future__ import annotations

import csv
import json
import os
import re
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Paths & runtime config
# ---------------------------------------------------------------------------

BOARD_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(BOARD_DIR, os.pardir))
RESULTS_DIR = os.path.realpath(
    os.environ.get("RESULTS_DIR", os.path.join(PROJECT_ROOT, "results", "bit-dqn"))
)
PORT = int(os.environ.get("PORT", "1942"))


# Legacy runs without an arch prefix duplicate the newer ``mlp_*`` /
# ``transformer_*`` variants and are excluded from the dashboard.
LEGACY_VARIANTS = frozenset({
    "basic_dqn",
    "her",
    "improved",
    "per",
    "double_dqn",
    "reward_shaping",
    "dueling_dqn",
    "transformer",
    "improved_transformer",
})


CONFIG_KEYS_OF_INTEREST = (
    "variant",
    "algorithm",
    "arch",
    "reward_mode",
    "double_dqn",
    "dueling",
    "per",
    "her",
    "her_ratio",
    "episodes",
    "eval_episodes",
    "max_sequence_length",
    "min_sequence_length",
    "hidden_dim",
    "transformer_heads",
    "transformer_layers",
    "transformer_dropout",
    "lr",
    "gamma",
    "batch_size",
    "buffer_size",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay_steps",
    "seed",
)


# ---------------------------------------------------------------------------
# Tiny YAML reader
# ---------------------------------------------------------------------------

_SCALAR_RE = re.compile(r"^[^#]*$")


def _parse_scalar(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return ""
    # strip surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s in ("null", "Null", "NULL", "~"):
        return None
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    # int
    try:
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
    except ValueError:
        pass
    # float
    try:
        if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", s):
            return float(s)
    except ValueError:
        pass
    return s


def _read_flat_yaml(path: str) -> Dict[str, Any]:
    """Minimal flat ``key: value`` YAML reader.

    The phd27fall config files are intentionally flat, so we sidestep the
    PyYAML dependency entirely.
    """
    out: Dict[str, Any] = {}
    if not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, _, rest = line.partition(":")
                key = key.strip()
                # strip inline comments
                rest = re.sub(r"\s+#.*$", "", rest)
                out[key] = _parse_scalar(rest)
    except OSError:
        return {}
    return out


# ---------------------------------------------------------------------------
# Filesystem walk
# ---------------------------------------------------------------------------

def _safe_listdir(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def _read_summary_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rows.append({
                        "sequence_length": int(r["sequence_length"]),
                        "success_rate": float(r["success_rate"]),
                        "bit_accuracy": float(r["bit_accuracy"]),
                        "prefix_accuracy": float(r["prefix_accuracy"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return []
    rows.sort(key=lambda x: x["sequence_length"])
    return rows


def _summarise_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n_points": 0,
            "mean_success": None,
            "mean_bit_accuracy": None,
            "mean_prefix_accuracy": None,
            "max_solved_len": None,
            "first_failure_len": None,
        }
    n = len(rows)
    mean_s = sum(r["success_rate"] for r in rows) / n
    mean_b = sum(r["bit_accuracy"] for r in rows) / n
    mean_p = sum(r["prefix_accuracy"] for r in rows) / n
    max_solved = None
    first_failure = None
    for r in rows:
        if r["success_rate"] >= 0.9:
            if max_solved is None or r["sequence_length"] > max_solved:
                max_solved = r["sequence_length"]
        else:
            if first_failure is None:
                first_failure = r["sequence_length"]
    return {
        "n_points": n,
        "mean_success": mean_s,
        "mean_bit_accuracy": mean_b,
        "mean_prefix_accuracy": mean_p,
        "max_solved_len": max_solved,
        "first_failure_len": first_failure,
    }


def _classify(config: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a few high-level tags from a raw config dict."""
    algo = str(config.get("algorithm", "")) or "unknown"
    arch_raw = str(config.get("arch", "")) or "unknown"
    if algo.startswith("baseline"):
        family = "baseline"
        arch = "baseline"
    else:
        family = "dqn"
        arch = arch_raw
    # The reward badge already conveys ``shaped`` vs ``sparse`` in the UI,
    # so ``features`` only carries the four DQN add-on flags to avoid the
    # ``shaped`` tag rendering twice on the same card.
    features: List[str] = []
    if bool(config.get("double_dqn")):
        features.append("double")
    if bool(config.get("dueling")):
        features.append("dueling")
    if bool(config.get("per")):
        features.append("PER")
    if bool(config.get("her")):
        features.append("HER")
    return {
        "family": family,
        "arch": arch,
        "arch_raw": arch_raw,
        "algorithm": algo,
        "reward_mode": str(config.get("reward_mode", "")),
        "double_dqn": bool(config.get("double_dqn")),
        "dueling": bool(config.get("dueling")),
        "per": bool(config.get("per")),
        "her": bool(config.get("her")),
        "features": features,
    }


def _slim_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {k: config.get(k) for k in CONFIG_KEYS_OF_INTEREST if k in config}


def _list_variants() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for name in _safe_listdir(RESULTS_DIR):
        if name in LEGACY_VARIANTS:
            continue
        variant_dir = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(variant_dir):
            continue
        config_path = os.path.join(variant_dir, "config.yaml")
        summary_csv = os.path.join(variant_dir, "summary.csv")
        examples_path = os.path.join(variant_dir, "examples.json")
        if not os.path.isfile(config_path):
            continue
        cfg = _read_flat_yaml(config_path)
        # fall back to directory name if config is missing variant
        cfg.setdefault("variant", name)
        rows = _read_summary_csv(summary_csv)
        summary = _summarise_metrics(rows)
        tags = _classify(cfg)
        items.append({
            "name": name,
            "config": _slim_config(cfg),
            "tags": tags,
            "summary": summary,
            "curve": rows,
            "has_examples": os.path.isfile(examples_path),
        })
    # ordering: family first (dqn then baseline), then arch, then name
    arch_order = {"transformer": 0, "transformer_encoder": 1, "mlp": 2, "baseline": 3, "unknown": 4}
    family_order = {"dqn": 0, "baseline": 1}
    items.sort(key=lambda it: (
        family_order.get(it["tags"]["family"], 9),
        arch_order.get(it["tags"]["arch"], 9),
        it["name"],
    ))
    return items


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _json_response(handler: SimpleHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _send_error(handler: SimpleHTTPRequestHandler, status: int, msg: str) -> None:
    body = msg.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------

_VARIANT_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _api_runs() -> Dict[str, Any]:
    variants = _list_variants()
    # The arch facet excludes baselines: baselines have no neural arch and are
    # already covered by the ``algorithm`` facet.
    archs = sorted({
        v["tags"]["arch"] for v in variants
        if v["tags"]["family"] != "baseline" and v["tags"]["arch"] not in {"baseline", "unknown"}
    })
    algos = sorted({v["tags"]["algorithm"] for v in variants})
    rewards = sorted({v["tags"]["reward_mode"] for v in variants if v["tags"]["reward_mode"]})
    return {
        "results_dir": RESULTS_DIR,
        "variants": variants,
        "facets": {
            "arch": archs,
            "algorithm": algos,
            "reward_mode": rewards,
        },
    }


def _api_examples(name: str) -> Dict[str, Any]:
    if not _VARIANT_NAME_RE.match(name):
        return {"error": "invalid variant name"}
    if name in LEGACY_VARIANTS:
        return {"error": "legacy variant excluded"}
    path = os.path.realpath(os.path.join(RESULTS_DIR, name, "examples.json"))
    if not path.startswith(RESULTS_DIR + os.sep) and path != RESULTS_DIR:
        return {"error": "path escapes RESULTS_DIR"}
    if not os.path.isfile(path):
        return {"error": "examples.json not found"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": f"failed to load: {exc}"}
    return {"variant": name, "examples": data}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server convention
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/runs":
            _json_response(self, _api_runs())
            return

        if path == "/api/examples":
            name = (query.get("variant") or [""])[0]
            if not name:
                _send_error(self, 400, "missing variant")
                return
            payload = _api_examples(name)
            status = 200
            if "error" in payload:
                status = 404 if "not found" in payload["error"] else 400
            _json_response(self, payload, status=status)
            return

        if path in ("/", ""):
            self.path = "/dashboard.html"
        elif path == "/dashboard":
            self.path = "/dashboard.html"
        return SimpleHTTPRequestHandler.do_GET(self)

    def log_message(self, format, *args):  # noqa: A002, N802 - http.server convention
        if args and isinstance(args[0], str):
            s = args[0]
            if "/api/" in s:
                return
        super().log_message(format, *args)


def main() -> None:
    os.chdir(BOARD_DIR)
    server = ThreadingHTTPServer(("", PORT), DashboardHandler)
    print(f"看板地址: http://21.6.253.14.devcloud.woa.com:{PORT}/dashboard.html")
    print(f"Results 目录: {RESULTS_DIR}")
    print(f"已排除 legacy variants: {sorted(LEGACY_VARIANTS)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
