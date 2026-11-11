# COMP9321 Assignment 3 — z5429763

电影收入回归与评级分类。需求详见 [REQUIREMENTS.md](REQUIREMENTS.md)，实际验证结果见 [VALIDATION.md](VALIDATION.md)。

## 环境

评分环境是 Python 3.13。建立自己的虚拟环境，并使用仓库提供的固定版本依赖：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

原始依赖文件末尾说明 `setuptools` 被 pip-compile 作为 unsafe 包省略。若需要运行完整 Jupyter/spaCy 环境，补装 `setuptools==82.0.1`；本提交脚本只使用标准库及清单中的 NumPy、pandas、CatBoost、LightGBM。

## 生成两份预测 CSV

```bash
python3 z5429763.py <train_file> <test_file>
```

`<train_file>` 和 `<test_file>` 是调用者提供的真实路径，可为绝对路径或相对路径。包含空格的路径需加引号。脚本会在**执行命令时的当前目录**写出：

- `z5429763_regression.csv`：`imdb_id,revenue`
- `z5429763_classification.csv`：`imdb_id,rating`

脚本独立训练收入回归的 CatBoost＋LightGBM 对数融合模型，以及评级分类的 CatBoost 模型。回归新增三个条件类别交叉和六类完整实体的拟合内留一频次聚合；分类特征和参数保持不变。每个模型使用 4 个 CPU 线程，无 GPU 或外部下载。目标列不参与特征生成，测试文件可以完全没有 `revenue`、`rating` 两列。输出文件会被本次运行覆盖。

## 重现 Notebook

Notebook 需要和 `z5429763.py` 放在一起，从仓库根目录执行。通过环境变量传入**同一对数据参数**：

```bash
export ASSIGNMENT_DATA_ARGS='"<train_file>" "<test_file>"'
jupyter nbconvert --to notebook --execute --inplace z5429763.ipynb \
  --ExecutePreprocessor.timeout=600
```

将上面尖括号部分替换为真实路径，保留变量内的双引号以支持路径中的空格。Notebook 比较多种配置，运行时间会长于最终提交脚本。

在云环境中，若用户主目录不可写，可先把缓存/配置位置设为可写目录：

```bash
export JUPYTER_DATA_DIR=/workspace/.cache/jupyter/data313
export JUPYTER_RUNTIME_DIR=/workspace/.cache/jupyter/runtime313
export JUPYTER_CONFIG_DIR=/workspace/.cache/jupyter/config313
export IPYTHONDIR=/workspace/.cache/ipython313
export MPLCONFIGDIR=/workspace/.cache/matplotlib313
export XDG_CACHE_HOME=/workspace/.cache
```

这些是开发工具缓存位置，不是模型或数据路径；正式脚本不需要这些变量。

## 测试

```bash
python -m pytest tests/test_submission.py -q
```

测试使用临时合成数据验证目标隔离、未知类别、缺失值、训练目标对预测的影响、任意输入路径、当前目录输出、列/行/ID/标签契约。完整提供数据上的 CLI 验证及性能见 `VALIDATION.md`。

正式提交前重新确认 CSE 上的总运行时间，Notebook 小于 3MB 且保留所有输出。提供测试集和云端时间不能保证隐藏测试集成绩或 CSE 时间。

题目要求 Notebook 保留全部执行输出，没有单独提交截图的要求。你仍需在 CSE 上确认正式脚本的总运行时间，并通过课程提交入口提交两份正式文件；GitHub 推送本身不等于课程提交。

## 回归改进与复核

当前回归方案在三个固定训练内划分中均优于前版融合模型，但尚未达到 RMSLE ≤1.7：

| 划分种子 | 前版融合模型 | 条件交叉＋留一实体频次 |
| --- | ---: | ---: |
| 42 | 1.864038 | **1.840865** |
| 2026 | 1.760112 | **1.738034** |
| 2027（方案固定后的额外检查） | 1.921720 | **1.916176** |

详细方法与局限见 [NEW_REGRESSION_EXPERIMENTS.md](NEW_REGRESSION_EXPERIMENTS.md)，最终正式命令验收见 [VALIDATION.md](VALIDATION.md)。前两划分已参与多轮开发，第三次增益较小，不能据此保证隐藏测试成绩。只读取训练文件的复核命令如下：

```bash
python experiments/regression_validation.py <train_file> --output /tmp/regression-results.json
```

该开发工具比较 `ReferenceRevenueRegressor` 与最终 `RevenueRegressor`，默认运行种子42、2026、2027三个固定划分及 Bootstrap；其运行时间不属于正式提交脚本的两分钟计时。Notebook 中包含同样的复核代码与输出。当前CLI耗时以 `VALIDATION.md` 为准，CSE仍需本人实测。

父模型的稳健性和预算缺失历史诊断见 [ROBUSTNESS_DIAGNOSTICS.md](ROBUSTNESS_DIAGNOSTICS.md)。可运行 `python experiments/robustness_validation.py <train_file> --output /tmp/robustness.json` 复核。那些拒绝结论对应新增交叉与留一特征之前的父模型，并非对新模型重新做过的降权/插补实验。

上传修改意见的历史评估见 [PROPOSAL_REVIEW.md](PROPOSAL_REVIEW.md)：该轮建议组合没有在两个既有划分同时改善，因此当时未替换父模型。后续独立设计的条件交叉＋完整实体留一频次才得到上表中的改善。可用 `experiments/proposal_validation.py` 重现旧消融；完整开发实验不属于正式脚本的两分钟计时。
