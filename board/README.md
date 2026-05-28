# Bit-DQN 实验看板

为 `results/bit-dqn/<variant>/` 下的 26 个有效 run 提供一个学术风、单页前端看板。
仿照 `Flow-Factory/ablation/sde_windows/board/` 的实现方式（纯 Python `http.server`
+ 单 HTML，无外部依赖、无构建步骤）。

## 启动

```bash
cd /mnt/shenzhen2cephfs/mm-base-vision/suzzetehu/phd27fall/board
export PORT=1948
python3 server.py
```

可选环境变量：

- `RESULTS_DIR`：默认指向 `../results/bit-dqn`。
- `PORT`：默认 `1942`。

启动后浏览器访问：

```
http://21.6.253.14.devcloud.woa.com:${PORT}/dashboard.html
```

## 数据约定

每个 variant 目录至少需要 `config.yaml` 与 `summary.csv`；`examples.json` 可选
（懒加载，按需取）。下列文件不会被前端使用：`summary.json`、`training_curves.json`。

```
results/bit-dqn/<variant>/
  config.yaml      # 分类 / 过滤维度
  summary.csv      # 列：bit_accuracy,prefix_accuracy,sequence_length,success_rate,variant
  examples.json    # 可选：{ "<sequence_length>": [{sample_id,target,generated,...}] }
```

### Legacy run 过滤

早期没有 arch 前缀、与新版 `mlp_*` / `transformer_*` 重复的 9 个 run 会在后端被硬过滤：

```
basic_dqn  her  improved  per  double_dqn  reward_shaping
dueling_dqn  transformer  improved_transformer
```

如需展示它们，编辑 `server.py` 中的 `LEGACY_VARIANTS` 集合。

## 功能

- **筛选（默认全展示）**
  - Arch 多选 pill：`mlp` / `transformer` / `transformer_decoder` / `baseline`（每个 arch 用
    不同 hue 标识，arch 在卡片头部以彩色 badge 永远展示）。
  - Algorithm、Reward mode 多选 pill。
  - Feature 三态 toggle：`double` / `dueling` / `PER` / `HER`，三态分别为
    *不限* / *只看包含* / *只看排除*。
  - variant 名称文本搜索。
- **分组**：按 `arch` / `algorithm` / `reward` 自动分章节展示；也可关掉分组。
- **每张 variant 卡片**
  - 左：关键 metric 表（mean success / bit-acc / prefix-acc，`max solved n (≥90%)`，
    `first failure n`）。
  - 右：自绘 SVG 折线图（`success / bit / prefix` 三个 metric 一键切换），颜色按 arch。
  - 底部：折叠的 examples 区，按 `sequence_length` 分桶；展开后懒加载
    `examples.json`，每个 sample 显示 target / generated（错位 bit 红色下划线），
    并统计该桶的命中率。
- **对比模式**
  - 每张卡左上角 checkbox，勾选 ≥ 2 个变体后点击「对比已选」会弹出
    lightbox，把所有选中曲线叠在同一张大图上，含 legend 与按 arch 配色（同一 arch
    内的多条线靠 HSL 亮度区分）。在 lightbox 顶部可切换 metric。
- 重置筛选、全选当前可见、清空勾选等便捷按钮在顶栏右侧。

## API（供调试 / 复用）

| 路径 | 返回 |
|---|---|
| `GET /api/runs` | `{results_dir, facets:{arch,algorithm,reward_mode}, variants:[{name,config,tags,summary,curve,has_examples}]}` |
| `GET /api/examples?variant=<name>` | `{variant, examples}`，懒加载该 variant 的 `examples.json` |

`curve` 来自 `summary.csv` 解析后的数组，每行：
`{sequence_length, success_rate, bit_accuracy, prefix_accuracy}`。
