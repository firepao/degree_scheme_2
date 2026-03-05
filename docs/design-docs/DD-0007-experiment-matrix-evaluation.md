# DD-0007: 主对比与消融实验矩阵 + HV/IGD 统计流程（P7）

- Status: Implemented
- Date: 2026-02-25
- Author: 杨振国（草案由 Copilot 协助整理）
- Scope: `scheml_2` P7（对应 DD-0001）

## 1. 背景与目标
P1-P6 已完成核心算法链路。下一步需要形成可答辩证据链：
1. 主对比（改进算法 vs baseline）
2. 消融矩阵（去原型交叉/去耦合变异/去动态精英/去代理）
3. 统一指标（HV/IGD/运行时间）
4. 可重复统计汇总（均值±标准差）

## 2. 问题定义与非目标
### 2.1 问题定义
实现两类能力：
- **批跑能力**：支持按配置开关生成对比与消融矩阵；
- **评估能力**：读取批次结果，计算每次运行与分组统计结果。

### 2.2 非目标
- 本次不加入论文最终排版图表脚本（P8）；
- 本次不做复杂统计显著性检验（先提供均值/标准差）。

## 3. 实验矩阵
默认因子：
- `engine`（默认 `deap_nsga2`）
- `seed`
- `population_size`
- `num_generations`
- `use_prototype_crossover`（T/F）
- `use_coupled_mutation`（T/F）
- `use_dynamic_elite_retention`（T/F）
- `surrogate.enabled`（T/F）

建议最小矩阵：
1. Main: 全模块开启
2. Ablation-A: 关闭原型交叉
3. Ablation-B: 关闭耦合变异
4. Ablation-C: 关闭动态精英
5. Ablation-D: 关闭代理模型

## 4. 指标定义
### 4.1 HV（Hypervolume）
- 最小化问题
- 参考点：全批次目标值最大值 + 边际
- 先使用 Monte Carlo 近似实现（可重复种子）

### 4.2 IGD（Inverted Generational Distance）
- 参考前沿：批次内所有运行目标集合的联合非支配前沿
- 每次运行到参考前沿的平均最近距离

### 4.3 运行时间
- 单次运行耗时（秒）
- 分组平均耗时与标准差

## 5. 设计细节
### 5.1 批跑脚本扩展
`experiments/batch_run.py` 增加布尔开关列表输入：
- `--prototype-flags`
- `--coupled-flags`
- `--dynamic-elite-flags`
- `--surrogate-flags`

并在 `manifest.csv` 写入：
- 四类开关状态
- 运行耗时 `elapsed_seconds`

### 5.2 评估脚本
新增 `experiments/evaluate_batch.py`：
- 输入：`--batch-root` 或 `--manifest`
- 输出：
  - `run_metrics.csv`（每次运行）
  - `summary_metrics.csv`（分组统计）

### 5.3 指标模块
新增 `src/fertopt/evaluation/metrics.py`：
- `nondominated_mask`
- `hypervolume_monte_carlo`
- `igd`

## 6. 验证方案
1. 单元测试：
- 指标函数输出形状和数值合法（非负）
- 非支配筛选正确
2. 集成验证：
- 小规模批跑生成 `manifest.csv`
- 评估脚本生成 `run_metrics.csv` 与 `summary_metrics.csv`

## 7. 风险与回滚
风险：
- HV Monte Carlo 近似存在方差
- 不同批次参考前沿不一致

缓解：
- 固定随机种子；
- 在同一批次内统一参考前沿和参考点。

回滚：
- 评估脚本可只输出 IGD 和时间，不阻塞主流程。

## 8. Implementation Plan
- [x] I1: 扩展配置开关并接入 runner
- [x] I2: 扩展 batch_run 生成消融矩阵与运行耗时
- [x] I3: 新增 evaluation/metrics.py 与测试
- [x] I4: 新增 evaluate_batch.py 汇总脚本
- [x] I5: 执行批跑 + 评估验证

## 9. 关联提交与结果
- Commit: 待补充
- Artifacts:
  - `artifacts/p7_batch_20260225_230932/`
  - `artifacts/p7_batch_20260225_230932/evaluation_20260225_231026/`
