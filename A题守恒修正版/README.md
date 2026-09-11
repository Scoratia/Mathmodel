# A题守恒修正版

先看《守恒修正与验证说明.md》。本目录包含修改后的模型、四份结果表、质量收支、场数据和已执行验证记录，不包含竞赛论文。

本版用固定材料单元干质量、实际水质量通量与守恒显热通量求解。后续真实湿密度由质量守恒确定，因此没有严格保留题给经验密度随含水率的整个函数；初始密度及题给比热、导热、扩散系数和输入数据保留。这一取舍需明确说明，不能视为已经完成材料实测验证。

- result1.xlsx：前1800秒温度、含水率。
- result2.xlsx：前3小时逐秒温度、含水率。
- result3.xlsx：固定尺寸条件下，每分钟含水率至严格达标的首个整分钟。
- result4.xlsx：按附件半径收缩，每分钟含水率、表面值和当前半径；材料外位置留空。

结果表为离线数值解，修改Excel单元格不会自动重算偏微分方程。需要改参数时修改求解设置后重跑。

Python依赖见requirements.txt。从本目录执行：

```text
python code/run_conservation_study.py --cache <临时缓存目录>
python code/check_insulated_drying.py
python code/make_report.py
```

导出Excel使用提供的build_excel.mjs和@oai/artifact-tool环境，将本目录绝对路径作为第一个参数、临时检查目录作为第二个参数。随后执行：

```text
python code/verify_exports.py
```

输入来源与旧版对照均保存在data目录，计算不依赖旧版文件夹。详细验证数据在results目录。COMSOL、CST及MATLAB未在本轮实际运行。
