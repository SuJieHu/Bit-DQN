# Bit-DQN 看板（GitHub Pages 静态版）

`../board/` 是开发用的 **Python 动态版**（`http.server` + 现读 `results/`）。
本目录是它的 **静态快照**，专门用于 GitHub Pages 公开展示，没有任何后端，
所有数据都被预先烤进 `api/*.json`。

```
board_github/
├── dashboard.html          # 与 ../board/dashboard.html 同源，仅改 2 处 fetch URL
├── build_static.py         # 生成 api/ 的脚本（复用 ../board/server.py 的逻辑）
└── api/                    # 由 build_static.py 产出，需要 commit 进 git
    ├── runs.json           # 对应原 /api/runs
    └── examples/           # 对应原 /api/examples?variant=<name>
        ├── mlp_basic_dqn.json
        └── ...
```

## 工作流

### 1. 重新生成静态数据（每次 `results/` 更新后跑一次）

```bash
cd /mnt/shenzhen2cephfs/mm-base-vision/suzzetehu/phd27fall
python3 board_github/build_static.py
```

只读 `../board/server.py`，不会改任何原文件。可选环境变量 `RESULTS_DIR`
覆盖默认路径（同 `server.py`）。

### 2. 本地预览（无需 server.py）

```bash
python3 -m http.server -d board_github 8000
# 然后浏览器打开 http://localhost:8000/dashboard.html
```

> 不能直接双击 `dashboard.html` 用 `file://` 打开——浏览器会拒绝 fetch
> 本地 JSON 文件（CORS）。必须经过任意 http 服务器。

### 3. 部署到 GitHub Pages

把 `board_github/` 一并 commit 到 GitHub，然后：

仓库 → **Settings** → **Pages** → **Source** 选 `Deploy from a branch` →
**Branch** `main` / **Folder** `/(root)` → Save。

等 1–2 分钟，访问：

```
https://<你的用户名>.github.io/<仓库名>/board_github/dashboard.html
```

如果想让网址更短，可以把 Source 的 Folder 改成 `/board_github`，但 Pages 只
认 `/(root)` 或 `/docs`，因此最干净的做法是：**把整个目录改名/复制成
`docs/`**，访问就变成 `https://<user>.github.io/<repo>/dashboard.html`。

## 与原 `../board/` 的差异

| 文件 | board/ | board_github/ |
|---|---|---|
| `server.py` | ✅ 必须，Python 后端 | ❌ 不需要 |
| `dashboard.html` | fetch `/api/runs` | fetch `api/runs.json` |
| 数据来源 | 启动时按需扫 `results/` | 预生成的 `api/*.json` |
| 是否能公开访问 | 否（内网 devcloud） | 是（GitHub Pages） |
| 何时更新 | 实时 | 重跑 `build_static.py` 后 |

原 `board/` 的所有交互功能（筛选、分组、对比 lightbox、examples 懒加载、
SVG 折线图等）在静态版完整保留——只是数据源从 HTTP API 换成了同源
JSON 文件，前端代码没有任何其它修改。

## 注意事项

- `api/runs.json` 中的 `results_dir` 字段被替换成 `"results/bit-dqn (static snapshot)"`，
  避免把开发机的绝对路径泄露到公网。
- `transformer_decoder_*` 系列若还没产出 `summary.csv` / `examples.json`，
  在静态版里会以"空卡片"形式出现；等训练完成 + 重跑 build 即可。
- `api/` 总体积约 2.6 MB（26 个 variant，18 个有 examples），远低于
  GitHub Pages 1 GB 仓库 / 100 GB 月流量上限。
