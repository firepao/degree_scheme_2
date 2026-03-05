# DD-0004: 协同/拮抗耦合变异（P4）设计与落地

- Status: Implemented
- Date: 2026-02-25
- Author: 杨振国（草案由 Copilot 协助整理）
- Scope: `scheml_2` P4（对应 DD-0001）

## 1. 背景与目标
P3 已实现原型引导交叉，但变异仍为通用多项式扰动，未体现 N-P-K 协同/拮抗先验。P4 的目标是把农学先验编码到变异阶段，实现“方向可解释 + 概率自适应”的变异机制。

## 2. 问题定义与非目标
### 2.1 问题定义
在 DEAP 主循环中引入耦合变异：
1. 定义协同/拮抗矩阵 `M`；
2. 在每个阶段按多元高斯扰动进行耦合更新；
3. 结合缺素偏离、阶段敏感度、多样性压力动态调整变异概率。

### 2.2 非目标
- 本次不实现动态精英保留（P5）；
- 本次不引入真实土壤诊断模型，仅实现统一可配置评分；
- 本次不改目标函数接口。

## 3. 方案与取舍
### 方案 A：保持 DEAP 默认变异
稳定但无先验知识。

### 方案 B：耦合高斯变异 + 动态概率（选择）
- 协方差中注入协同/拮抗先验；
- 动态概率提升收敛期扰动能力；
- 保留边界投影与数值稳定修复。

## 4. 设计细节
### 4.1 新增算子模块
新增 `src/fertopt/operators/mutation.py`：
- `build_synergy_antagonism_matrix(...)`
- `dynamic_mutation_probability(...)`
- `coupled_mutation(...)`

### 4.2 耦合扰动
对每个阶段向量 $z_t=[N_t,P_t,K_t]$：
$$
\Delta z_t \sim \mathcal{N}(0, \sigma^2 I + \alpha D_t M D_t)
$$
其中 `D_t` 根据阶段施肥尺度构建，协方差矩阵做对称化与对角抖动确保可采样。

### 4.3 动态变异概率
$$
p_t = p_0 \cdot [1 + \beta\cdot Def + \gamma\cdot Sens + \delta\cdot DivPress]
$$
并做区间裁剪 `p_t in [0, p_max]`。

### 4.4 运行时接入
在 `runner.py` 的 DEAP 循环中：
1. 计算种群 `DivPress`；
2. 对每个个体根据阶段敏感度随机选择阶段并计算 `p_t`；
3. 命中后执行 `coupled_mutation`，否则保持原样。

## 5. 配置扩展
在 `mutation` 下新增：
- `sigma_base`
- `alpha_knowledge`
- `rho_np`, `rho_nk`, `rho_pk`
- `p_max`
- `stage_sensitivity`（长度=`num_stages`）

## 6. 验证方案
1. 算子测试：
   - 输出不越界；
   - 概率在区间内；
   - 协方差采样稳定。
2. 集成测试：
   - 主循环与产物不受破坏；
   - `pytest` 全通过。

## 7. 风险与回滚
风险：
- 协方差非正定导致采样失败；
- 动态概率过高导致震荡。

缓解：
- 采用抖动修复（diagonal jitter）；
- 设 `p_max` 上限和参数默认值。

回滚：
- 保留 `random_search` 和既有主循环；
- 可通过将 `alpha_knowledge=0` 退化为独立噪声。

## 8. Implementation Plan
- [x] I1: 新增 `operators/mutation.py`
- [x] I2: 扩展 `MutationConfig` 与默认配置
- [x] I3: 在 `runner.py` 接入耦合变异与动态概率
- [x] I4: 新增测试用例
- [x] I5: 执行 `pytest` 与 baseline 验证

## 9. 关联提交与结果
- Commit: 待补充
- Artifacts: `artifacts/baseline_deap_nsga2_20260225_214610/`
