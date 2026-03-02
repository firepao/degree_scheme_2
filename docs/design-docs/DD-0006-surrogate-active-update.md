# DD-0006: LightGBM 代理模型与主动更新机制（P6）

- Status: Implemented
- Date: 2026-02-25
- Author: 杨振国（草案由 Copilot 协助整理）
- Scope: `scheml_2` P6（对应 DD-0001）

## 1. 背景与目标
P1-P5 已完成数据/目标冻结与进化算子主干落地。当前优化过程仍默认直接调用真实目标函数，尚未引入代理模型来降低评估成本。P6 目标是接入 LightGBM 代理，并提供每 `G` 代主动更新机制。

## 2. 问题定义与非目标
### 2.1 问题定义
实现 `SurrogateManager`：
1. 对目标子集（默认 `yield`, `nitrogen_loss`）训练 LightGBM 回归器；
2. 在主循环中用代理预测替代对应真值目标评估；
3. 每 `G` 代选择 `B` 个样本做真值查询并增量重训。

### 2.2 非目标
- 本次不实现复杂不确定性估计（如深度集成/贝叶斯）；
- 本次不接入外部 DSSAT API，仅保留真值回调接口；
- 本次不改动 P1 冻结的目标函数命名与输出方向。

## 3. 方案与取舍
### 方案 A：全真值评估
准确但代价高，不利于后续扩展。

### 方案 B：代理 + 主动更新（选择）
- 优点：可降低评估成本，且通过周期性真值回填控制偏差；
- 成本：需要维护训练缓存与重训流程。

## 4. 设计细节
### 4.1 新增模块
新增 `src/fertopt/models/surrogate.py`：
- `SurrogateManager.initialize(X, y_true)`
- `SurrogateManager.predict_objectives(X, problem)`
- `SurrogateManager.active_update(generation, X_candidates, true_eval_fn)`

### 4.2 主动更新策略
- 触发条件：`generation % update_interval_g == 0`
- 查询策略：基于“到已标注样本最近距离”的覆盖性选样（远离历史样本优先）
- 查询批量：`query_batch_size`
- 更新动作：追加真值样本并重训目标子模型

### 4.3 主循环接入
在 `deap_nsga2` 中：
1. 初始化种群先做一次真值评估并初始化代理；
2. 适应度计算改为 `SurrogateManager.predict_objectives`；
3. 每 G 代执行 `active_update`，并刷新种群适应度。

## 5. 配置扩展
`surrogate` 增加：
- `enabled`（bool）
- `target_objectives`（list）
- `model_num_estimators`
- `model_learning_rate`

保留已有：
- `update_interval_g`
- `query_batch_size`

## 6. 验证方案
1. 单元测试：
   - 初始化训练成功；
   - 预测输出维度正确；
   - `active_update` 后训练样本数量增长。
2. 集成测试：
   - 关闭代理时行为与当前一致；
   - 开启代理时主循环可跑通。

## 7. 风险与回滚
风险：
- LightGBM 未安装导致运行失败；
- 代理偏差导致搜索漂移。

缓解：
- 在 `enabled=True` 时做依赖检查并给清晰报错；
- 保留定期真值回填策略；
- 默认配置可先关闭代理。

回滚：
- `surrogate.enabled=false` 即退回全真值评估。

## 8. Implementation Plan
- [x] I1: 新增 `models/surrogate.py`
- [x] I2: 扩展 `SurrogateConfig` 与 `default.yaml`
- [x] I3: 在 `runner.py` 接入代理评估与每 G 代主动更新
- [x] I4: 新增测试（有 lightgbm 时执行）
- [x] I5: 执行 `pytest` 与 baseline 验证

## 9. 关联提交与结果
- Commit: 待补充
- Artifacts: `artifacts/baseline_deap_nsga2_20260225_224105/`
