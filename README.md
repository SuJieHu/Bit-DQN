# DQN Bit-Sequence Generation
> Experinents Results Board:
https://sujiehu.github.io/dqn/board_github/dashboard.html

Token-by-token DQN for a simple bit-sequence imitation task: given a random
target bit sequence of length `n ∈ [1, 50]`, the agent emits the sequence
one bit at a time and receives an exact-match reward at the end of the
episode. We sweep `n` and train one model per `n`, then compare a vanilla
DQN against the standard "5 tricks" (reward shaping, Double DQN, Dueling,
PER, HER) across three Q-network backbones (MLP, Transformer encoder,
full encoder-decoder Transformer).

---

## 1. Installation

The code is tested with **Python 3.10+** and **PyTorch 2.7 (CUDA 12.8)**,
but any reasonably recent PyTorch build will work; you can edit the
PyTorch version line in `requirements.txt` to match your CUDA.

```bash
# 1) Create an environment (conda or venv, your choice)
conda create -n dqn python=3.10 -y
conda activate dqn

# 2) Install dependencies
pip install -r requirements.txt
```

---

## 2. Repository layout

```text
.
├── README.md                      # this file
├── requirements.txt
│
├── configs/                       # one YAML per experiment
│   ├── default.yaml               #   base config inherited by everything
│   ├── baseline_random.yaml       #   random-policy baseline (no training)
│   ├── baseline_oracle.yaml       #   oracle baseline (upper bound)
│   ├── mlp/                       #   MLP backbone
│   ├── transformer_encoder/       #   encoder-only Transformer
│   └── transformer/               #   full seq2seq Transformer (enc+dec)
│
├── src/                           # library code
│   ├── config.py                  #   YAML loader, supports `extends:` chains
│   ├── env.py                     #   gymnasium BitSequenceEnv
│   ├── agent.py                   #   DQNAgent (online+target nets, HER, …)
│   ├── trainer.py                 #   trains one variant across all n
│   ├── baselines.py               #   random / oracle baselines
│   ├── utils.py                   #   seeding, metrics, IO
│   ├── models/                    #   Q-network architectures
│   │   ├── base.py
│   │   ├── mlp_q.py
│   │   ├── dueling_q.py
│   │   └── transformer_q.py       #   encoder-only AND full encoder-decoder
│   └── replay/                    #   replay buffers (uniform, prioritized)
│
├── scripts/
│   ├── train.py                   # single entrypoint, --config <name>
│   └── runs/                      # one launcher script per experiment
│       ├── _common.sh             #   sourced by every NN_*.sh
│       ├── launch_all.sh          #   spawn all 23 experiments in parallel
│       ├── 01_baseline_random.sh
│       ├── 02_baseline_oracle.sh
│       ├── 03_mlp_basic_dqn.sh
│       ├── … 26_transformer_improved.sh
│       └── logs/<timestamp>/<NN_name>.log
│
└── results/<run_name>/<variant>/  # produced by the trainer
```

---

## 3. Running a single experiment

The trainer is fully driven by a YAML config; the CLI only exposes
meta-flags (run name, wandb on/off, debug overrides). The `--config`
argument is a path **relative to `configs/`** (the `.yaml` suffix is
optional).

```bash
# Baselines (no training, just evaluation)
python scripts/train.py --config baseline_random
python scripts/train.py --config baseline_oracle

# MLP backbone
python scripts/train.py --config mlp/basic_dqn        # Task 1 control
python scripts/train.py --config mlp/reward_shaping
python scripts/train.py --config mlp/double_dqn
python scripts/train.py --config mlp/dueling_dqn
python scripts/train.py --config mlp/per
python scripts/train.py --config mlp/her
python scripts/train.py --config mlp/improved         # all 5 tricks

# Transformer-encoder backbone (same 7 variants)
python scripts/train.py --config transformer_encoder/basic_dqn
python scripts/train.py --config transformer_encoder/improved
# …

# Full seq2seq Transformer backbone (same 7 variants)
python scripts/train.py --config transformer/basic_dqn
python scripts/train.py --config transformer/improved
# …
```

Useful CLI flags (all optional; everything else lives in the YAML):

| flag                       | purpose                                                  |
| -------------------------- | -------------------------------------------------------- |
| `--run-name NAME`          | name of the results subdirectory (default `bit-dqn`)     |
| `--results-dir DIR`        | root directory for outputs (default `results/`)          |
| `--device {cpu,cuda,auto}` | override `device:` from the YAML                         |
| `--seed N`                 | override the random seed                                 |
| `--use-wandb` / `--no-wandb` | force-enable or force-disable wandb logging            |
| `--wandb-project NAME`     | override wandb project name                              |
| `--wandb-entity NAME`      | override wandb entity                                    |
| `--max-sequence-length N`  | smoke-test override (e.g. `--max-sequence-length 4`)     |
| `--episodes N`             | smoke-test override (episodes per `n`)                   |
| `--eval-episodes N`        | smoke-test override (eval episodes per `n`)              |

Quick 1-minute smoke test:

```bash
python scripts/train.py --config mlp/basic_dqn \
  --max-sequence-length 4 --episodes 80 --eval-episodes 40 --no-wandb
```

---

## 4. Running all experiments

`scripts/runs/` contains one shell wrapper per experiment, plus a
`launch_all.sh` that spawns all 23 jobs in parallel. Each wrapper just
sources `_common.sh` (which activates conda and sets default flags) and
then calls `scripts/train.py` with the right config and the right
`CUDA_VISIBLE_DEVICES`.

**Before running these scripts you must edit `scripts/runs/_common.sh`**:

- `REPO_ROOT` and `CONDA_SH` are hard-coded absolute paths — change them
  to match your machine, or delete those lines and source your own conda
  setup.
- The HTTP/HTTPS proxy lines are specific to our cluster — delete them
  unless you actually need them.
- The `WANDB_API_KEY` fallback should be **removed or replaced with your
  own key** before publication.

Run one experiment via its wrapper:

```bash
bash scripts/runs/03_mlp_basic_dqn.sh
```

Run all 23 in parallel (one log file per script under
`scripts/runs/logs/<timestamp>/`):

```bash
bash scripts/runs/launch_all.sh

---

## 5. Outputs

For every variant, the trainer writes a single results folder:

```text
results/<run_name>/<variant>/
├── config.yaml             # the *resolved* (post-extends) config, for reproducibility
├── summary.json / .csv     # one row per n: success_rate, bit_accuracy, prefix_accuracy, …
├── training_curves.json    # per-n training history (episode reward, loss, eval metrics)
├── training_curves.png     # all n overlaid on one figure
├── success_vs_n.png        # success / bit-acc / prefix-acc vs n
├── examples.json           # qualitative generated-vs-target samples, keyed by n
└── models/n<k>.pt          # one checkpoint per n
```

Nothing is written per-`n` into its own directory; every quantity that
varies with `n` is aggregated into a single file per variant.

If wandb is enabled (`use_wandb: true` in the YAML, or `--use-wandb` on
the CLI), the trainer also logs:

- per-`n` training panels: `train/n<k>/episode_reward`, `train/n<k>/loss`, …
- periodic eval metrics: `eval/n<k>/success_rate`, …
- a single headline curve `length_curve/success_rate` with
  `length_curve/n` as the x-axis.

---

## 6. Configuration system

Configs use a very small `extends:` mechanism (see `src/config.py`).
`default.yaml` defines every key; every other file inherits from it (or
from a `_base.yaml` in its subfolder) and overrides only what differs.
The trainer freezes the final, fully-resolved config to
`results/.../config.yaml` for reproducibility. Any CLI flag you pass to
`scripts/train.py` overrides the corresponding YAML field.

Key fields you might want to change (all live in `configs/default.yaml`):

- `max_sequence_length` / `min_sequence_length` — range of `n` to sweep over.
- `episodes`, `eval_episodes`, `eval_interval` — training schedule per `n`.
- `arch` — one of `mlp`, `transformer_encoder`, `transformer`.
- `reward_mode` — `sparse` (Task 1) or `shaped`.
- `double_dqn`, `dueling`, `per`, `her`, `her_ratio` — algorithmic switches.
- `lr`, `gamma`, `batch_size`, `buffer_size`, `target_update_interval`,
  `epsilon_*` — standard DQN hyperparameters.
- `use_wandb`, `wandb_project`, `wandb_entity` — logging.

---

## 7. License

Released for academic use; please cite the project if you build on it.
