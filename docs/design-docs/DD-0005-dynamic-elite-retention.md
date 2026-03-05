# DD-0005: 动态精英保留（P5）设计与落地

- Status: Implemented
- Date: 2026-02-25
- Author: 杨振国（草案由 Copilot 协助整理）
- Scope: `scheml_2` P5（对应 DD-0001）

## 1. 背景与目标
P4 已实现协同/拮抗耦合变异，当前环境选择仍主要依赖标准 NSGA-II 选择。P5 的目标是引入“动态精英保留”：将非支配等级与局部稀疏度融合，平衡收敛性与多样性。

## 2. 问题定义与非目标
### 2.1 问题定义
在每代环境选择中：
1. 计算非支配等级 `F_i`；
2. 计算综合距离（目标空间 + 决策空间）；
3. 计算局部稀疏度 `D_i`；
4. 构建动态精英得分并筛选下一代。

### 2.2 非目标
- 本次不实现 HV/IGD 指标逻辑（后续 E 阶段）；
- 本次不改动目标函数定义和代理模型接口。

## 3. 公式与策略
### 3.1 综合距离
$$
Dist_{ij} = \omega_f ||f_i-f_j||_2 + \omega_x ||x_i-x_j||_2
$$
其中 $\omega_f + \omega_x = 1$。

### 3.2 局部稀疏度
$$
D_i = \frac{1}{k}\sum_{j\in N_k(i)} Dist_{ij}
$$

### 3.3 动态精英得分
$$
E_i = \frac{1}{F_i} + \alpha(t)\cdot \frac{D_i}{\max_j D_j}
$$
$$
\alpha(t)=\alpha_0\exp(-\beta t/T_{max})
$$

## 4. 设计细节
### 4.1 新增模块
新增 `src/fertopt/operators/selection.py`：
- `dynamic_elite_select_indices(...)`
- 内部实现非支配排序、综合距离与稀疏度计算。

### 4.2 Runner 接入
在 DEAP 主循环中，将 `population + offspring` 的环境选择替换为动态精英筛选。

### 4.3 配置扩展
新增 `selection` 配置：
- `alpha0`
- `beta_decay`
- `omega_f`
- `omega_x`
- `k_neighbors`

## 5. 验证方案
1. 单元测试：
   - 返回索引数量正确且不重复；
   - 分数计算稳定（无 NaN/Inf）。
2. 集成测试：
   - 主循环可跑通且产物完整；
   - `pytest` 全通过。

## 6. 风险与回滚
风险：
- 过强稀疏度权重导致收敛变慢；
- 邻居数设置不合理导致排序抖动。

缓解：
- 提供默认保守参数；
- 权重退火确保后期收敛。

回滚：
- 可切回标准 `selNSGA2`（保留代码路径）。

## 7. Implementation Plan
- [x] I1: 新增 `operators/selection.py`
- [x] I2: 扩展 `config` 与 `default.yaml`
- [x] I3: 在 `runner.py` 接入动态精英保留
- [x] I4: 新增测试
- [x] I5: 执行 `pytest` 与 baseline

## 8. 关联提交与结果
- Commit: 待补充
- Artifacts: `artifacts/baseline_deap_nsga2_20260225_221013/`
