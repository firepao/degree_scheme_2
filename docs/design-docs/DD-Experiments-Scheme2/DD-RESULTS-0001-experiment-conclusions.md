# DD-RESULTS-0001: Degree Scheme 2 实验阶段性结论（消融 + 参数敏感性）

- Status: Implemented (Interim)
- Date: 2026-03-08
- Author: 杨振国（草案由 Copilot 协助整理）
- Scope: `scheml_2` 对比实验、消融实验、核心参数敏感性实验的阶段性结果总结

## 1. 背景与目标
本文件用于记录 `degree_scheme_2` 当前阶段实验的正式结论，服务于后续设计迭代、论文撰写与答辩材料整理。

当前实验工作分为三部分：
1. **对比实验（Comparative Experiments）**：方案2 与外部 5 个最新多目标算法的比较；
2. **消融实验（Ablation Studies）**：分析 PCX、Coupled Mutation、Dynamic Elite 等模块的贡献；
3. **核心参数敏感性实验（Parameter Sensitivity）**：分析关键参数变化对 HV / IGD 的影响。

本文件仅对**当前已经产出完整汇总结果**的实验做正式总结；对于仍在运行的实验，仅给出状态说明，不给出最终排名结论。

## 2. 结果使用前提与解释边界
### 2.1 指标说明
本轮实验采用以下指标：
- **HV（Hypervolume）**：越大越好，用于衡量帕累托解集覆盖能力；
- **IGD（Inverted Generational Distance）**：越小越好，用于衡量当前解集相对参考前沿的逼近程度。

### 2.2 结果解释边界
根据当前实验脚本实现，参考前沿由**同一批次全部运行结果的联合非支配前沿**构建。因此：
1. **同一批次内部**的各方法对比是有效的；
2. **不同批次之间**不宜直接比较绝对 HV / IGD 数值；
3. 若某一实验尚未完成，则其参考前沿仍可能变化，对最终结论有影响。

因此，本文件中的所有正式结论均限定在**当前已完成批次内部**，避免跨批次、跨未完成集合做过度推断。

## 3. 对比实验（Comparative Experiments）正式结论
### 3.1 输出位置
对比实验结果当前输出至（本轮历史运行）：
- 汇总 CSV：`/mnt/d/opencode_repo/artifacts/comparison_benchmark/summary_metrics.csv`
- 单次运行 CSV：`/mnt/d/opencode_repo/artifacts/comparison_benchmark/run_metrics.csv`
- 可视化图：`/mnt/d/opencode_repo/artifacts/comparison_benchmark/pareto_front_comparison.png`

后续脚本默认输出位置已修正为：
- `artifacts/comparison_benchmark_<timestamp>/`

对应日志文件命名为：
- `artifacts/comparison_benchmark_<timestamp>/comparison_benchmark_<timestamp>.log`

### 3.2 汇总结果
| 方法 | HV Mean ± Std | IGD Mean ± Std |
|---|---:|---:|
| scheme2 | 12.42M ± 10.91M | 1391.59 ± 899.46 |
| nsga3 | 18.36M ± 10.53M | 1390.01 ± 703.56 |
| moead | 0.00M ± 0.01M | 7192.77 ± 270.50 |
| agemoea | 17.55M ± 10.79M | **1266.91 ± 707.22** |
| ctaea | 9.01M ± 6.27M | 1494.24 ± 814.04 |
| rvea | 0.74M ± 0.93M | 5544.69 ± 1105.87 |
| smsemoa | **19.99M ± 11.45M** | 1474.27 ± 747.77 |

### 3.3 主要发现
#### 结论 A：从 HV 看，当前最优方法为 `smsemoa`
- `smsemoa` 的 HV 均值最高（19.99M），说明其在当前评估口径下具有最强的帕累托覆盖能力；
- `nsga3` 与 `agemoea` 也表现出较高的 HV，整体优于 `scheme2`；
- `moead` 与 `rvea` 的 HV 极低，说明其在当前问题设置下表现明显不适配。

#### 结论 B：从 IGD 看，当前最优方法为 `agemoea`
- `agemoea` 的 IGD 均值最低（1266.91），表明其在当前批次内相对最接近联合参考前沿；
- `scheme2` 与 `nsga3` 的 IGD 非常接近，但均未优于 `agemoea`；
- `moead` 与 `rvea` 的 IGD 远高于其他方法，进一步说明其在本问题上的逼近质量较差。

#### 结论 C：`scheme2` 当前尚未在主对比实验中取得领先
- 当前 `scheme2` 在 HV 上明显落后于 `smsemoa`、`nsga3` 和 `agemoea`；
- 在 IGD 上，`scheme2` 仅接近 `nsga3`，但仍不优于 `agemoea`；
- 因此，在当前实验范围内，**不能宣称方案2已经优于外部最新 baseline**。

#### 结论 D：主对比实验整体波动较大，结论需结合标准差谨慎解释
- 多个方法的 HV / IGD 标准差都较大，说明不同 seed 下波动明显；
- `scheme2` 本身的 HV 和 IGD 波动也较大，表明稳定性尚不足；
- 因此更适合将当前主对比结论描述为“阶段性排名结果”，而非“已被完全验证的最终结论”。

### 3.4 正式结论表述
建议在正式文档/论文中使用如下表述：
> 在当前已完成的主对比实验中，方案2尚未表现出对外部先进 baseline 的稳定优势。从 HV 指标看，SMS-EMOA 的覆盖能力最佳；从 IGD 指标看，AGE-MOEA 的前沿逼近质量最佳。方案2当前在 IGD 上与 NSGA-III 接近，但整体仍未形成明确领先地位。因此，现阶段更适合将方案2定位为“仍需继续优化与验证的候选方法”，而非“已经全面优于现有基线的方法”。

## 4. 消融实验（Ablation Studies）正式结论
### 4.1 输出位置
- 汇总 CSV：`artifacts/ablation_benchmark/summary_metrics.csv`
- 单次运行 CSV：`artifacts/ablation_benchmark/run_metrics.csv`
- 可视化图：`artifacts/ablation_benchmark/pareto_front_ablation.png`

### 4.2 汇总结果
| 变体 | HV Mean ± Std | IGD Mean ± Std |
|---|---:|---:|
| Full_Scheme2 | 36.17M ± 0.87M | 130.87 ± 79.65 |
| w/o_Coupled_Mutation | 36.07M ± 1.04M | 97.39 ± 31.91 |
| w/o_Dynamic_Elite | 35.75M ± 1.59M | 94.73 ± 44.32 |
| w/o_PCX_(use_SBX) | **36.68M ± 0.60M** | 103.19 ± 61.99 |
| Baseline_NSGA-II | 35.99M ± 1.00M | 94.92 ± 14.77 |

### 4.3 主要发现
#### 结论 A：完整方案 `Full_Scheme2` 当前并未表现出全面最优
- 从 **HV** 看，最佳设置并不是完整方案，而是 `w/o_PCX_(use_SBX)`；
- 从 **IGD** 看，完整方案反而是当前最差的一组；
- 因此，当前阶段**不能**得出“Full_Scheme2 在消融实验中整体最优”的结论。

#### 结论 B：PCX 模块在当前实验范围内未体现出稳定正贡献
- 去掉 PCX 并改用 SBX 后，HV 均值最高，且标准差最小；
- 这提示从帕累托覆盖能力角度看，PCX 当前版本未显示出优于 SBX 的稳定趋势；
- 因此，PCX 至少在当前实现与参数设置下，**尚未被当前实验结果充分支持为稳定有效模块**。

#### 结论 C：Dynamic Elite 模块可能对前沿逼近质量存在条件依赖或负面影响
- 去掉 Dynamic Elite 后，IGD 均值最优；
- 这提示当前 Dynamic Elite 机制对解集向参考前沿的稳定逼近可能存在条件依赖；
- 尤其在完整方案中，IGD 波动明显偏大，表明其稳定性存在不足。

#### 结论 D：Coupled Mutation 的贡献相对温和，当前证据尚不支持其带来稳定优势
- 去掉 Coupled Mutation 后，HV 变化不大，IGD 反而略好；
- 这提示 Coupled Mutation 当前更像是“影响搜索行为”的调节模块，而非已被当前实验充分支持的核心增益模块。

### 4.4 稳定性分析
当前消融实验中，`Baseline_NSGA-II` 的 IGD 标准差最小（14.77），稳定性最好；
而 `Full_Scheme2` 的 IGD 标准差最大（79.65），且单次运行中存在明显退化值。

这说明：
> 当前完整方案的主要问题不是“完全无效”，而是**稳定性不足、模块间协同关系尚未完全理顺**。

### 4.5 正式结论表述
建议在正式文档/论文中使用如下表述：
> 消融实验结果显示，在当前实验设置与评价口径下，方案2中的多个模块尚未表现出一致且稳定的正向增益。特别是 PCX 与 Dynamic Elite 模块，在当前实现下未同时改善 HV 与 IGD，反而体现出覆盖能力与前沿逼近质量之间的权衡关系。因此，方案2当前更适合被描述为“具有潜力的结构设计”，而非“已被完全验证的稳定最优结构”。

## 5. 参数敏感性实验（Parameter Sensitivity）正式结论
### 5.1 输出位置
- 汇总 CSV：`artifacts/parameter_sensitivity/summary_metrics.csv`
- 单次运行 CSV：`artifacts/parameter_sensitivity/run_metrics.csv`
- 可视化图：`artifacts/parameter_sensitivity/sensitivity_*.png`

### 5.2 参数一：`population_size`
| 取值 | HV Mean ± Std | IGD Mean ± Std |
|---|---:|---:|
| 40 | 30.87M ± 2.70M | 355.96 ± 121.76 |
| 80 | 33.87M ± 1.40M | 257.89 ± 100.51 |
| 120 | 35.87M ± 1.72M | 208.49 ± 117.18 |
| 160 | **36.37M ± 0.74M** | **171.74 ± 30.60** |

**正式结论：**
- 随着种群规模增大，HV 持续提升，IGD 持续下降；
- `population_size = 160` 同时取得当前最优均值和较低波动；
- 因此在当前实验范围内可以认为：
> 在当前问题设置与所测区间内，增大种群规模能够稳定提升方案2的搜索质量与结果稳定性。

### 5.3 参数二：`beta_strength_k`
| 取值 | HV Mean ± Std | IGD Mean ± Std |
|---|---:|---:|
| 2.0 | 28.42M ± 1.44M | 775.19 ± 245.56 |
| 4.0 | 30.44M ± 1.70M | 462.61 ± 162.16 |
| 8.0 | 33.70M ± 0.62M | 252.39 ± 100.06 |
| 12.0 | **36.28M ± 1.61M** | **165.23 ± 46.13** |

**正式结论：**
- `beta_strength_k` 增大时，性能改善趋势非常明显；
- 特别是 IGD 呈现持续下降趋势，说明更强的初始化偏置有助于前沿逼近；
- 因此在当前实验范围内可以认为：
> 更强的 Beta 偏置初始化策略对当前优化问题是有效的，`beta_strength_k = 12.0` 是当前阶段的最优候选值。

### 5.4 参数三：`crossover.gamma0`
| 取值 | HV Mean ± Std | IGD Mean ± Std |
|---|---:|---:|
| 0.01 | **34.46M ± 1.32M** | **191.86 ± 67.18** |
| 0.05 | 34.31M ± 1.75M | 306.09 ± 51.43 |
| 0.1 | 34.02M ± 1.51M | 378.71 ± 135.46 |
| 0.2 | 33.74M ± 0.98M | 242.57 ± 75.77 |

**正式结论：**
- `gamma0` 并不存在极强的单调趋势；
- 但从当前均值表现看，较小的 `gamma0` 更优；
- 当前最优候选值为 `0.01`。

**建议表述：**
> 在当前所测区间内，原型牵引强度不宜过大。当前实验显示，小幅度牵引更有利于保持搜索质量，而过强的牵引可能削弱解集逼近能力。

### 5.5 参数四：`selection.alpha0`
| 取值 | HV Mean ± Std | IGD Mean ± Std |
|---|---:|---:|
| 0.0 | 33.69M ± 1.13M | 337.08 ± 214.88 |
| 0.2 | 33.27M ± 1.53M | **256.48 ± 77.16** |
| 0.5 | 33.38M ± 1.50M | 293.10 ± 104.34 |
| 0.8 | **34.12M ± 2.07M** | 264.43 ± 87.91 |

**正式结论：**
- `alpha0` 对 HV 和 IGD 的影响不一致；
- `0.8` 在 HV 上最好，但波动最大；
- `0.2` 在 IGD 上最好，稳定性相对更可接受；
- 因此它更像是一个“搜索覆盖 vs 前沿逼近”的权衡参数。

**建议表述：**
> `selection.alpha0` 不存在单一绝对最优值，其合理设置取决于实验目标：若更重视解集覆盖能力，可倾向更高取值；若更重视前沿逼近质量，则较低取值更合适。

## 6. 综合结论
### 6.1 当前已经成立的结论
1. **在当前实验范围内，种群规模增大对方案2有明确正向作用**，当前最优候选值为 `population_size = 160`；
2. **在当前实验范围内，更强的 Beta 偏置初始化是有效的**，当前最优候选值为 `beta_strength_k = 12.0`；
3. **`crossover.gamma0` 更适合较小取值**，当前最优候选值为 `0.01`；
4. **`selection.alpha0` 体现明显的 HV / IGD 权衡关系**，需要按目标导向设定。

### 6.2 当前尚不能成立的结论
1. 不能宣称 `Full_Scheme2` 已经在消融实验中整体优于所有变体；
2. 不能宣称 PCX、Dynamic Elite、Coupled Mutation 均已被实验充分证明有效；
3. 不能宣称方案2已经优于外部 5 个最新 baseline，因为当前主对比实验结果并未显示其取得领先。

### 6.3 当前阶段的总体判断
综合来看，当前实验最有价值的信息是：
> 方案2的**参数配置方向已经比较清晰**，但其**结构模块设计仍需进一步验证与优化**。

换言之，现阶段更适合将方案2描述为：
> “参数潜力已经得到实验支持，但结构层面的协同增益仍需进一步打磨与验证。”

## 7. 后续建议（面向下一轮实验）
1. 等待主对比实验全部完成后，补充外部 baseline 的最终结论；
2. 针对 PCX 与 Dynamic Elite 模块，进一步排查其在 IGD 上退化的原因；
3. 在下一轮正式实验中优先采用以下参数候选：
   - `population_size = 160`
   - `beta_strength_k = 12.0`
   - `crossover.gamma0 = 0.01`
   - `selection.alpha0 = 0.2` 或 `0.8`（视重点指标而定）

## 8. 关联产出位置
- 主控文档：`docs/design-docs/DD-Experiments-Scheme2/README.md`
- 本结论文档：`docs/design-docs/DD-Experiments-Scheme2/DD-RESULTS-0001-experiment-conclusions.md`
- 消融实验结果：`artifacts/ablation_benchmark_<timestamp>/`（后续默认输出）
- 参数敏感性结果：`artifacts/parameter_sensitivity_<timestamp>/`（后续默认输出）
- 对比实验结果：`artifacts/comparison_benchmark_<timestamp>/`（后续默认输出）
- 实验日志结果：各自时间戳目录下的同名前缀日志文件（如 `comparison_benchmark_<timestamp>.log`）
