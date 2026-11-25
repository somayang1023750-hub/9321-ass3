# 已保存的软分区回归候选

候选名称：`soft_hurdle_v1`。保存日期：2026-11-25。最初保存于9634c08，后于2816b72接入正式脚本。用户随后在CSE实测255.06秒，要求简化；当前正式脚本已改用去掉整体模型的精简软分区。本文记录原始完整候选及其权重，不代表最新配置。最新指标见[CURRENT_MODEL_METRICS.md](CURRENT_MODEL_METRICS.md)，简化依据见[COMPACT_RUNTIME.md](COMPACT_RUNTIME.md)。

## 保存的是哪一个模型

训练标签以收入≤1000、>1000分组，两个区间分别拟合回归器。门控分类器只用输入特征预测低收入概率p，区间预测按p与1-p在log1p收入空间软混合，再与当前总模型各占50%。没有用测试文件真实收入做路由。

此前硬路由分区没有改善，因此不作为此保存版本。源码为`experiments/soft_hurdle_model.py`，历史证据为`experiments/soft_hurdle_validation_results.json`。

| 历史验证 | 总模型OOF RMSLE | 软混合OOF RMSLE |
|---|---:|---:|
|3折CV重复1|1.867265|1.856081|
|3折CV重复2|1.868142|1.858423|

6个验证折均改善，两次全量OOF的均值约1.8677→1.8573。每折拟合约10669行，和80/20划分的12803行不同，不能把不同划分的绝对分数直接比较。结果仍高于1.7；重复使用相同训练数据，不等于独立外部测试。这些验证对应本文完整候选，不作为精简模型的CV成绩。2816b72完整候选后来在CSE实测255.06秒，超出要求。

## train、validation和test的区别

| 名称 | 数据量 | 用途 |
|---|---:|---|
|课程提供的train.csv|16004条|有标签，用来开发和拟合模型|
|训练内validation|80/20实验中3201条|从train.csv中按索引暂时留出；对应模型只拟合另外12803条，用来选择和比较方案|
|课程提供的test.csv|2287条|独立提供的评估数据；即使含标签，也不用于训练、预处理拟合、选特征或调参|
|老师隐藏测试集|未知|评分时由老师提供，真实标签不可用于我们的建模|

验证集没有另存为CSV，所以目录中看不到validation.csv。`experiments/discussion_validation.py`的compare函数里有train_test_split；保存候选的`experiments/soft_hurdle_validation.py`里有StratifiedKFold。验证时，各折模型不接触其验证标签。最终保存权重时，再用完整16004条训练数据重拟合，这是不同于验证的交付步骤。

此前汇报的1.8308等是训练内验证分数，不是模型对自身拟合样本的训练误差，也不是提供test.csv或隐藏集分数。群聊的“train1.76/test1.85”未说明train指拟合误差还是训练内验证，不能直接与我们的数字比较。拟合误差较低、独立评估误差较高是常见现象。

## 群聊中能确认的有用信息

较清晰截图提示运行时长应留余量，机器负载可能导致波动；低收入难例影响大，与我们已有误差分解一致。100秒是同学建议的余量目标，原题仍是两个任务合计120秒。“CSE只有两核”是同学说法，尚未核对具体执行机器，不据此改线程设置。“老师肯定会放宽阈值”也是推测，须以老师正式公告为准。低分或数据脏不能成为改标签、删验证难例的依据。

后续用户提供Desktop.zip原图，已完成分段和关键段落核查，没有发现新的可复现优化方法。原题确有“无人达到原阈值时可能放宽”的条件性条款；群聊未提供已经公布的新标准。截图中的27人满分是Assignment 2，不是ASS3。

## 权重、加载与复现

最初保存的本机权重位于`artifacts/soft_hurdle_v1/soft_hurdle.joblib`，同目录manifest.json记录当时的训练数据/源码/权重SHA256、参数、软件版本和重载一致性检查。manifest的源码哈希对应9634c08保存时的版本，不是后来正式脚本的哈希；历史证据不覆盖改写。配套源码与权重已另存下载归档。权重文件由已有*.joblib忽略规则保存在工作区；源码、配置和验证记录纳入版本控制。正式CLI从传入训练文件重训，不加载历史权重。

```bash
python experiments/save_soft_hurdle.py <train_file> --output-dir <new_directory>
```

已有保存目录不会被覆盖。脚本只读取传入的训练CSV，不打开test.csv。保存时的预测检查只验证目标盲性和序列化一致性，不作为验证成绩。历史OOF分数属于各折模型，并非全量拟合权重的训练分数。

在仓库根目录、匹配Python3.13和依赖版本下加载本机权重：

```python
import joblib
import z5429763 as solution
model = joblib.load('artifacts/soft_hurdle_v1/soft_hurdle.joblib')
log_predictions = model.predict(input_frame)
revenue_predictions = solution.revenue_from_log(log_predictions)
```

重现训练内验证，所有模型从各自拟合分区重新学习：

```bash
python experiments/soft_hurdle_validation.py <train_file> --output <report.json> --threads 2
```

这会训练许多模型，开发验证时间不等于正式两任务CLI的计时。此命令复核的是历史完整候选。当前精简版同时减少分类树数，两份预测CSV重新生成；原始数据和依赖未修改，最新指标另行记录。
