# 目标函数规范冻结（P1）

- Version: v1.0
- Date: 2026-02-25
- Scope: `scheml_2/src/fertopt/core/objectives.py`
- 对应设计文档: `DD-0001`

## 1. 目标函数集合
当前默认三目标（最小化形式）：
1. `yield`：产量目标（通过取负号转最小化）
2. `cost`：施肥成本目标
3. `nitrogen_loss`：氮损失目标

配置入口：`configs/default.yaml -> objectives`

## 2. 当前实现口径（占位版）
当前实现是“可替换模拟函数”，用于跑通优化闭环，后续会替换为“代理模型 + 真值评估”。

### 2.1 `yield`
设每阶段养分向量为 $z_t=[N_t,P_t,K_t]$，当前近似：
$$
productivity=\sum_t\sum_{j\in\{N,P,K\}}\sqrt{\max(z_{t,j},0)}-\lambda\cdot\mathrm{mean}((N_t-P_t)^2)
$$
$$
yield(x)=-productivity
$$
其中惩罚系数 $\lambda=0.001$。

### 2.2 `cost`
$$
cost(x)=\sum_t (p_NN_t+p_PP_t+p_KK_t)
$$
当前默认价格向量：
- $p_N=5.2$
- $p_P=6.8$
- $p_K=4.5$
（单位建议：CNY/kg）

### 2.3 `nitrogen_loss`
$$
loss=0.06\sum_tN_t+0.01\sum_t\max(N_t-0.5(P_t+K_t),0)
$$
$$
nitrogen\_loss(x)=loss
$$

## 3. 冻结约束
1. 目标函数名称冻结为：`yield`, `cost`, `nitrogen_loss`；
2. 所有目标输出均为“越小越好”；
3. 目标函数输入维度必须等于 `num_stages * len(nutrients)`；
4. 后续替换真实目标时，函数名不变，内部逻辑可替换。

## 4. 未来替换计划（P6 对齐）
- `yield`：替换为 `surrogate_yield(x)`，并周期性用真值校正；
- `nitrogen_loss`：替换为 `surrogate_nloss(x)` 或过程模型输出；
- `cost`：保留解析式，价格从配置读取（避免硬编码）。

## 5. 配置化建议（下一步）
建议新增配置项：
- `objective_params.cost.price_npk`
- `objective_params.yield.balance_penalty`
- `objective_params.nloss.base_coeff`
- `objective_params.nloss.excess_coeff`

## 6. 验收清单（P1）
- [x] 冻结目标函数名称与方向
- [x] 冻结当前占位公式说明
- [x] 冻结后续替换边界（接口不变）
- [ ] 将系数全面外置到配置（P2/P6）
