# DD-0009: 复现脚本与答辩演示材料打包（P9）

- Status: Implemented
- Date: 2026-02-25
- Author: 杨振国（草案由 Copilot 协助整理）
- Scope: `scheml_2` P9（对应 DD-0001）

## 1. 背景与目标
P1-P8 已完成算法实现、实验矩阵、评估汇总与论文图表自动导出。当前剩余任务是“交付层闭环”：
1. 让评审可一键复现实验核心结果；
2. 将论文与答辩所需材料按标准目录打包；
3. 保证结果、脚本、图表、结论可追溯。

## 2. 问题定义与非目标
### 2.1 问题定义
新增“复现与打包”流程：
- 一键复现脚本（环境检查 -> 批跑 -> 评估 -> 图表与总表导出）
- 答辩材料目录生成脚本（图、表、关键结论、版本信息）
- 最终交付包（zip）

### 2.2 非目标
- 本次不实现 Web 可视化平台；
- 本次不新增算法功能；
- 本次不做外部云端自动部署。

## 3. 交付物定义
### 3.1 复现脚本
建议文件：
- `scripts/reproduce_all.ps1`（Windows 主入口）
- `scripts/reproduce_all.py`（跨平台逻辑）

输出：
- `artifacts/repro_<ts>/`
  - `batch_*/`
  - `evaluation_*/`
  - `paper_*/`
  - `repro_manifest.json`

### 3.2 答辩材料目录
建议目录：
- `deliverables/defense_<ts>/`
  - `slides_assets/`（答辩图）
  - `tables/`（总表与消融表）
  - `key_findings.md`（关键结论）
  - `method_overview.md`（方法摘要）
  - `version_info.txt`（环境与git信息）

### 3.3 最终打包
- `deliverables/repro_package_<ts>.zip`
- `deliverables/defense_package_<ts>.zip`

## 4. 流程设计
### 4.1 一键复现流程
1. 检查 Python/Conda 环境与依赖版本；
2. 执行批跑（主对比 + 消融最小矩阵）；
3. 执行评估（HV/IGD/耗时）；
4. 导出论文图表与结果总表；
5. 写入 `repro_manifest.json`（记录命令、时间戳、输入输出路径）。

### 4.2 答辩材料生成流程
1. 从 `paper_*` 目录收集图表；
2. 从总表生成“可口播”摘要（Top 结果、消融影响）；
3. 复制到 `defense_<ts>` 目录并生成目录说明。

## 5. 环境锁定与可追溯
建议新增：
- `environment.lock.txt`（`pip freeze`）
- `run_commands.log`（执行命令）
- `git_state.txt`（分支/commit hash/changed files）

目标：确保“论文图表中的每个数字都能追溯到某次批跑与评估目录”。

## 6. 验证方案
1. 复现验证：
- 在新终端执行一次 `reproduce_all`，成功生成完整 `repro_<ts>`。
2. 材料验证：
- `defense_<ts>` 目录包含图、表、结论、版本文件。
3. 完整性验证：
- 随机抽查 2 个图和 2 个表，确认来源路径可追溯。

## 7. 风险与回滚
风险：
- 依赖版本差异导致结果波动；
- 路径参数写死导致复现失败。

缓解：
- 统一入口参数与相对路径；
- 输出 manifest 与 lock 文件；
- 对关键步骤失败即停并提示。

回滚：
- 若一键流程失败，可按 P7/P8 脚本分步执行。

## 8. Implementation Plan
- [x] I1: 新增复现入口脚本（PowerShell + Python）
- [x] I2: 新增答辩材料收集与目录生成脚本
- [x] I3: 新增版本锁定与运行日志输出
- [x] I4: 新增压缩打包脚本
- [x] I5: 端到端演练并回写 DD-0001 P9

## 9. 关联提交与结果
- Commit: 待补充
- Artifacts:
  - Repro: `artifacts/repro_i5_20260226_104113/`
  - Defense: `deliverables/defense_20260226_104143/`
  - Packages:
    - `deliverables/repro_package_20260226_104328.zip`
    - `deliverables/defense_package_20260226_104328.zip`
    - `deliverables/package_manifest_20260226_104328.json`
