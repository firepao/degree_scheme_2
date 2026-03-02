# DD-0008: 论文图表自动导出与结果总表生成（P8）

- Status: Implemented
- Date: 2026-02-25
- Author: 杨振国（草案由 Copilot 协助整理）
- Scope: `scheml_2` P8（对应 DD-0001）

## 1. 背景与目标
P7 已完成批量实验矩阵与评估汇总（HV/IGD/耗时），但论文写作仍需要手动整理图表与表格。P8 的目标是将“实验结果 -> 论文素材”自动化，确保结果可复现、可追溯、可直接用于论文与答辩材料。

## 2. 问题定义与非目标
### 2.1 问题定义
新增两类脚本：
1. **图表导出脚本**：从 `run_metrics.csv` 与 `summary_metrics.csv` 自动生成论文级图表；
2. **结果总表脚本**：汇总主对比与消融结果，输出一份总表（CSV/Markdown）。

### 2.2 非目标
- 本次不负责 Word/LaTeX 排版模板；
- 本次不做统计显著性检验扩展（如 Wilcoxon/Friedman）；
- 本次不改动优化算法本体。

## 3. 输入与输出约定
### 3.1 输入
- `artifacts/<batch_name>/manifest.csv`
- `artifacts/<batch_name>/evaluation_<ts>/run_metrics.csv`
- `artifacts/<batch_name>/evaluation_<ts>/summary_metrics.csv`

### 3.2 输出
在 `artifacts/<batch_name>/paper_<ts>/` 下生成：
- `fig_hv_boxplot.png`（HV 分组箱线图）
- `fig_igd_boxplot.png`（IGD 分组箱线图）
- `fig_runtime_bar.png`（平均运行时间柱状图）
- `fig_pareto_overlay.png`（多运行帕累托叠加图）
- `table_main_ablation.csv`（主对比与消融总表）
- `table_main_ablation.md`（便于论文拷贝）
- `paper_manifest.json`（记录输入文件与脚本参数）

## 4. 脚本设计
### 4.1 图表导出脚本
建议文件：`experiments/export_paper_figures.py`

功能：
- 读取 `run_metrics.csv` 与 `summary_metrics.csv`
- 根据分组键（engine + 各开关）自动生成标签
- 统一风格输出图（分辨率、尺寸、字体）
- 支持 `--top-n` / `--group-by` / `--dpi` 参数

### 4.2 结果总表脚本
建议文件：`experiments/build_result_report.py`

功能：
- 读取 `summary_metrics.csv`
- 生成“主模型 vs 消融”对照表
- 输出字段建议：
  - 组名
  - HV_mean ± HV_std
  - IGD_mean ± IGD_std
  - Time_mean ± Time_std
  - runs
- 生成 CSV + Markdown 双格式

## 5. 图表与表格规范（论文友好）
1. 文件命名固定、可重复生成；
2. 坐标轴单位清晰（HV/IGD/seconds）；
3. 图例标签与实验配置一致；
4. 所有图表来自同一批次评估，避免跨批次混用；
5. 输出目录中保存参数与输入文件快照（manifest）。

## 6. 验证方案
1. 脚本功能验证：
- 在现有 `p7_batch` 上可一键生成所有图表与总表；
- 输出文件数量与命名符合约定。
2. 内容一致性验证：
- 总表数值与 `summary_metrics.csv` 一致；
- 图表数据抽样核对至少 2 组。

## 7. 风险与回滚
风险：
- 分组标签过多导致图表可读性差；
- 不同批次字段不一致导致脚本失败。

缓解：
- 提供 `--group-by` 与 `--filter` 参数；
- 做字段存在性检查与错误提示。

回滚：
- 保留 P7 的原始 `run_metrics.csv` / `summary_metrics.csv` 作为最终兜底。

## 8. Implementation Plan
- [x] I1: 新增 `export_paper_figures.py`
- [x] I2: 新增 `build_result_report.py`
- [x] I3: 统一图表风格与输出目录规范
- [x] I4: 在 `p7_batch` 上跑通并产出 `paper_<ts>`
- [x] I5: 回写 DD 状态与路线图 P8

## 9. 关联提交与结果
- Commit: 待补充
- Artifacts:
  - `artifacts/p7_batch_20260225_230932/paper_demo/`
  - `artifacts/p7_batch_20260225_230932/paper_demo/paper_manifest.json`
