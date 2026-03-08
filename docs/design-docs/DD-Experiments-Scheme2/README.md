# Degree Scheme 2: 实验方案执行总览 (Experiment Execution Master Doc)

## 1. 任务背景 (Background)
本设计文档专为 `degree_scheme_2` 的**对比实验**、**消融实验**和**核心参数讨论**建立。所有实验均要求运行五次取均值与标准差(Mean & Std)，并且输出评估指标(HV, IGD 等)的 CSV 文件，同时生成可视化的帕累托前沿(Pareto Front)散点图。

## 2. 目录结构与进度追踪 (Directory & Progress)

### 2.1 阶段一：对比实验 (Phase 1: Comparative Experiments)
*目标：挑选最新的5个先进 MOEA 算法作为外部 Baseline。*
- [x] Baseline 接口调研与框架设计
- [x] 算法 1 接口实现: MOEA/D 
- [x] 算法 2 接口实现: AGE-MOEA 
- [x] 算法 3 接口实现: C-TAEA
- [x] 算法 4 接口实现: RVEA
- [x] 算法 5 接口实现: SMS-EMOA
- [x] 实验评估脚本 (5-seed 运行机制) 
- [x] CSV及可视化输出

### 2.2 阶段二：消融实验 (Phase 2: Ablation Studies)
*目标：优化并完善已有的消融实验框架。*
- [x] 消融配置审计与重构
- [x] 5-seed 消融运行脚本
- [x] CSV及可视化输出

### 2.3 阶段三：核心参数讨论 (Phase 3: Parameter Sensitivity)
*目标：挑选 2-4 个核心参数进行网格化测试和分析。*
- [x] 确定评估的核心参数及取值梯度 (`population_size`, `beta_strength_k`, `crossover.gamma0`, `selection.alpha0`)
- [x] 5-seed 参数测试运行脚本
- [x] 趋势图与误差线生成

---

## 3. 输出汇总位置 (Artifacts Tracking)
*(随着各个环节实现逐步填写该区域)*

- **阶段性结论文档**: `docs/design-docs/DD-Experiments-Scheme2/DD-RESULTS-0001-experiment-conclusions.md`

- **Baselines 代码路径**: `/mnt/d/opencode_repo/degree_scheme_2/scheml_2/src/fertopt/baselines/external/`
- **对比实验评估脚本**: `experiments/run_comparison_benchmark.py`
- **消融实验评估脚本**: `experiments/run_ablation_study.py`
- **参数讨论评估脚本**: `experiments/run_parameter_sensitivity.py`
- **CSV结果保存路径**:
  - 对比实验: `artifacts/comparison_benchmark_<timestamp>/` 下的 `run_metrics.csv` 和 `summary_metrics.csv`
  - 消融实验: `artifacts/ablation_benchmark_<timestamp>/` 下的 `run_metrics.csv` 和 `summary_metrics.csv`
  - 参数敏感度: `artifacts/parameter_sensitivity_<timestamp>/` 下的 `run_metrics.csv` 和 `summary_metrics.csv`
- **图表输出路径**:
  - 对比实验: `artifacts/comparison_benchmark_<timestamp>/pareto_front_comparison.png`
  - 消融实验: `artifacts/ablation_benchmark_<timestamp>/pareto_front_ablation.png`
  - 参数敏感度: `artifacts/parameter_sensitivity_<timestamp>/sensitivity_*.png`
- **日志输出路径**:
  - 对比实验: `artifacts/comparison_benchmark_<timestamp>/comparison_benchmark_<timestamp>.log`
  - 消融实验: `artifacts/ablation_benchmark_<timestamp>/ablation_benchmark_<timestamp>.log`
  - 参数敏感度: `artifacts/parameter_sensitivity_<timestamp>/parameter_sensitivity_<timestamp>.log`
