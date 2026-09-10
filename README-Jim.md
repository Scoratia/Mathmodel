# A题建模成果使用说明

先阅读 **建模与验证说明.md**，再查看四个 result 文件。这里交付模型、求解结果和验证证据，不包含竞赛论文。

## 结果

- 第 3 问临界时间 57.47400912 h，按整分钟操作为 57 h 29 min。
- 第 4 问临界时间 51.08851315 h，按整分钟操作为 51 h 6 min。
- 第 1、2 问的全部指定时空输出分别在 result1.xlsx、result2.xlsx。
- 第 3、4 问全程每分钟输出在 result3.xlsx、result4.xlsx。
- 第 4 问空白表示该固定位置已在收缩后的药材外，不是缺失测量，也不是零含水率。

数值按四位小数显示。末行中心值可能显示 0.1500，实际未舍入值分别为 0.1499902244、0.1499787721，满足严格小于 0.15。时长和数值均以说明中的有效传质、环境延拓和均匀径向收缩假设为条件。

## 复现计算

使用 Python 3.12。在本目录打开终端，建议创建独立虚拟环境后安装 requirements.txt。

```text
python -m pip install -r requirements.txt
python code/run_study.py
python code/check_geometry.py
python code/make_figures.py
python code/make_report.py
```

完整计算包括五档网格、解析基准、两个时间积分器和 24 个敏感性/结构情景。计算时间随设备变化。本次已提供完整输出，不必重新运行才能阅读结果。默认缓存写入 cache/，可以通过 `--cache` 指定另一缓存目录。缓存仅用于同一份模型代码；修改模型、输入或数值算法后应使用新的缓存目录。`--quick` 仅用于环境检查，会得到粗网格输出，不能覆盖正式结果后当作最终解。

主求解仅依赖 NumPy 与 SciPy。Matplotlib 用于图；openpyxl 只用于读取已导出的 Excel 并独立核对，不参与数值求解或 Excel 创作。输入已保存成 JSON，不依赖本机原始题包路径。

## Excel 重新导出

已附四个完整 Excel。代码中的数值复现将重建 results/result1.json 至 result4.json 等数据。

若在具有 `@oai/artifact-tool` 的 Codex 文档运行环境中需要重新生成 Excel，执行：

```text
node --max-old-space-size=8192 code/build_excel.mjs . cache/excel_qa
python code/verify_deliverables.py
```

独立 Python 环境可以直接读取 JSON/NPZ 结果；Excel 导出程序另外需要上述 JavaScript 库。不是只运行 Python 就会更新 Excel。模板存放于 data/templates/。

## 验证范围

results/validation.json：解析基准、网格变化、温度与浓度边界、模型水分收支、参数扰动、停机判据和题设展示表。

results/geometry_validation.json：40×40 网格二维有限圆柱的端面影响。

results/delivery_checks.json：读取实际 Excel 后逐格核对的结果和文件摘要。

figures/：5 张静态结果图。code/make_figures.py 在 Windows 优先使用微软雅黑，其次黑体或 Noto Sans CJK SC；其他系统需安装相应中文字体。

注意：附件未给药材内部温湿度或称重观测。这里完成的是数值检验和假设敏感性分析，不能据此宣称真实实验预测误差或保证获奖等级。半径的留出拟合误差也不能代替温湿度的实验验证。

## 参数和采样的修改位置

code/model.py 的 Settings 控制扩散系数倍率、换热传质系数、后期边界、半径模式、网格和积分容差。

code/run_study.py 的 payload 和采样时间数组控制输出窗口。result2.xlsx 采用题目表 3、4 的前 3 h 窗口；全程模型已求解至第 3 问终点。如收到官方补充说明要求全程每秒输出，可直接扩展采样范围。

原始输入单位、边界解释、潜热情景和有效密度的物理限制均在建模与验证说明中详述。调整这些假设后应重新运行和验证，不能只改结论中的数字。
