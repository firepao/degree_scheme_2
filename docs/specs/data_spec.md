# 数据规范冻结（P1）

- Version: v1.0
- Date: 2026-02-25
- Scope: `scheml_2` 方案二（施肥多目标优化）
- 对应设计文档: `DD-0001`

## 1. 目标与原则
本规范用于冻结实验数据接口，确保后续算法替换（DEAP、创新算子、代理模型）不改变数据输入输出口径。

原则：
1. 先冻结接口，再替换数据源；
2. 变量命名、单位、维度统一；
3. 缺失值与切分规则可复现；
4. 所有实验目录必须可追溯到相同数据规范版本。

## 2. 样本单位与主键
- 样本单位：`plot-season`（地块-季节）
- 主键建议：`sample_id`
- 推荐辅助字段：
  - `site_id`
  - `year`
  - `crop`
  - `treatment_id`

## 3. 决策变量（优化输入）
设生育阶段数为 `T`（由 `configs/default.yaml` 的 `num_stages` 给定），每阶段 3 个养分：N/P/K。

决策向量：
$$
X=[N_1,P_1,K_1,\dots,N_T,P_T,K_T]
$$

字段命名（扁平格式）：
- `N_s1`, `P_s1`, `K_s1`, ..., `N_sT`, `P_sT`, `K_sT`

单位与范围：
- 单位：`kg/ha`
- 默认边界：`[0, 300]`（由配置 `var_lower_bound/var_upper_bound` 控制）

## 4. 目标变量（监督/评估标签）
必须至少包含以下三类（可用观测值或模拟真值）：
1. `yield_obs`：产量（建议单位 `kg/ha` 或 `t/ha`，必须在数据字典注明）
2. `cost_obs`：成本（建议单位 `CNY/ha`）
3. `n_loss_obs`：氮损失（建议单位 `kg N/ha`）

若使用 DSSAT 或其他过程模型：
- 对应真值字段命名为 `yield_true`, `n_loss_true`；
- 代理模型预测字段命名为 `yield_pred`, `n_loss_pred`。

## 5. 上下文字段（建议）
用于代理模型与可解释分析，不直接作为决策变量：
- 土壤：`soil_ph`, `soil_om`, `soil_tn`, `cec`
- 气象：`temp_mean`, `rain_sum`, `rad_sum`
- 地理：`lat`, `lon`, `alt`
- 管理：`irrigation_level`, `plant_density`

## 6. 缺失值处理规则（冻结）
1. 列级阈值：缺失率 > 50% 的字段默认删除；
2. 数值型：优先中位数填补（并记录是否填补标志）；
3. 类别型：填补为 `Unknown`；
4. 标签字段（`yield_obs/cost_obs/n_loss_obs`）不允许缺失，缺失样本剔除。

## 7. 切分规则（冻结）
- 默认划分：`train/val/test = 75% / 12.5% / 12.5%`
- 固定随机种子：来自配置 `seed`
- 若存在明显年份泄漏风险，优先按年份分层切分（后续在具体实验 DD 冻结）。

## 8. 文件与目录规范
建议数据目录：
- `data/raw/`：原始数据（只读）
- `data/processed/`：清洗后的主表
- `data/splits/`：切分索引

建议最小文件：
- `data/processed/dataset_v1.csv`
- `data/processed/data_dictionary_v1.md`
- `data/splits/split_seed42_v1.json`

## 9. 验收清单（P1）
- [x] 冻结输入向量结构与命名
- [x] 冻结三目标标签口径与单位要求
- [x] 冻结缺失值处理与切分规则
- [ ] 落地真实数据主表与数据字典（P1.1 后续）
