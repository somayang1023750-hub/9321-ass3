# 当前精简模型：Train / Val / Test（2026-11-25）

正式CLI使用CompactHurdleRevenueRegressor与CompactRatingClassifier。用户实测上一完整软混合版2816b72在CSE耗时255.06秒后，本版去掉整体收入模型，并减少两个专家、门控与评级分类器的树数。算法与选型依据见[COMPACT_RUNTIME.md](COMPACT_RUNTIME.md)。

配置和源码先冻结，再全量重训、写出提供Test预测、最后评分；没有用Test标签选择配置。参数与原始指标见experiments/current_model_metrics_freeze.json和experiments/current_model_metrics.json。

## 最新指标

RMSLE越低越好，Macro F1与Accuracy越高越好。

| 数据与口径 | 样本数 | 回归RMSLE | 分类Macro F1 | 分类Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Train：开发拟合 | 12,803 | 1.457950 | 0.880433 | 0.940483 |
| Val：开发留出 | 3,201 | 1.840734 | 0.768874 | 0.884099 |
| Train：全量重训 | 16,004 | 1.483474 | 0.875552 | 0.938015 |
| Test：课程提供 | 2,287 | 1.941241 | 0.738150 | 0.868824 |

前两行属于seed42按rating分层的80/20开发划分，同一组模型拟合12803条、验证3201条。后两行属于另一组最终模型，先用全部16004条Train重训，再预测全量Train与2287条提供Test。Train是拟合样本分数，不是OOF或隐藏测试成绩；其预测使用全拟合映射，与训练时交叉拟合表示也有区别。老师隐藏测试集指标未知。

## 相对完整软混合的取舍

| 指标 | 2816b72完整软混合 | 当前精简版 |
| --- | ---: | ---: |
| Val RMSLE | 1.826482 | 1.840734 |
| Val Macro F1 | 0.775005 | 0.768874 |
| 提供Test RMSLE | 1.930587 | 1.941241 |
| 提供Test Macro F1 | 0.745810 | 0.738150 |

主划分Val回归误差上升约0.0143，分类下降约0.0061；这是降低耗时的取舍。精简版提供Test分类低于0.74满分阈值约0.00185，回归仍未达到1.7。不因看到这些Test结果而回头调参数。之前完整模型的指标与计时另存full_hurdle_metrics.json、full_hurdle_metrics_freeze.json、full_hurdle_cli.json。

固定精简配置后，seed2026与2718的Val RMSLE为1.730329、1.903840，Macro F1为0.757723、0.756137。三个划分都来自Train，存在重叠与开发复用。原完整候选的两次3折CV均值1.857252不属于本精简模型。

关于Train差距：当前开发Train/Val回归差约0.383，原完整模型约0.404；差距略缩小，但Val误差略升。因此不能把差距变小解释成泛化更好或过拟合已解决。群聊train1.76的口径未说明，不能直接与我们的拟合样本分数比较。训练侧编码诊断见COMPACT_RUNTIME.md。

## 复现

```bash
python experiments/current_model_metrics.py <train_file> <test_file> \
  --output /tmp/current-metrics.json --prediction-dir /tmp/current-predictions
```

此开发审计分别训练开发模型与全量模型，耗时不等于一次正式CLI。Notebook第11节内嵌同一逻辑并保留真实输出，前面历史章节保留各阶段原始输出。正式提交只依赖z5429763.py与已执行的z5429763.ipynb，不读取预存权重或experiments模块。

本版完整CLI云端实测49.457秒、exit0，相对完整软混合118.944秒减少约58.4%。35项测试及输出契约见VALIDATION.md。旧版CSE255.06秒不能作为本版耗时；本版仍需CSE复测。
