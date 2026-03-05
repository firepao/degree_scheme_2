# DD-0002: DEAP NSGA-II 主循环替换设计

- Status: Implemented
- Date: 2026-02-25
- Author: 杨振国（草案由 Copilot 协助整理）
- Scope: `scheml_2` P2（对应 DD-0001）

## 1. 背景与目标
当前 `BaselineRunner` 仍以 `random_search` 为主，`geatpy_nsga2` 在 Windows 环境存在安装阻塞，影响稳定复现与后续创新算子接入。

本设计目标：
1. 以 DEAP 实现稳定可复现的 NSGA-II 主循环；
2. 保持现有 `Runner` / `experiment` 接口兼容；
3. 不改变产物规范（CSV + Pareto 图 + manifest）；
4. 为 P3/P4/P5 创新算子接入预留扩展点。

## 2. 问题定义与非目标
### 2.1 问题定义
在 `src/fertopt/core/runner.py` 中新增 `deap_nsga2` 引擎，并替换原 `geatpy_nsga2` 路径，使优化主循环由 DEAP 执行。

### 2.2 非目标
- 本次不实现原型引导交叉、协同拮抗耦合变异、动态精英保留；
- 本次不接入真实数据和代理模型训练逻辑；
- 本次不改变目标函数定义（沿用 P1 冻结接口）。

## 3. 方案对比与取舍
### 方案 A：继续使用 geatpy
- 优点：模板接入快；
- 缺点：当前 Windows 环境安装失败，影响团队复现。

### 方案 B：切换 DEAP（选择）
- 优点：跨平台稳定、依赖易安装、算子定制灵活；
- 缺点：需要手写部分主循环逻辑。

结论：选择 DEAP 作为 P2 主引擎。

## 4. 设计细节
### 4.1 引擎接口
- `BaselineRunner.run(..., engine=...)` 支持：
  - `random_search`（保留，作为回滚/对照）
  - `deap_nsga2`（新增主引擎）

### 4.2 NSGA-II 主循环要点
- 个体编码：实数向量，维度 `num_stages * len(nutrients)`；
- 初始种群：沿用 Beta 偏向初始化；
- 交叉：`cxSimulatedBinaryBounded`；
- 变异：`mutPolynomialBounded`；
- 选择：`selNSGA2` + `selTournamentDCD`；
- 边界：全程按配置上下界裁剪；
- 适应度：通过 `FertilizationProblem.evaluate` 统一计算。

### 4.3 兼容与输出
- 不改 `run_baseline.py` 与 `batch_run.py` 的输出协议；
- 继续输出：
  - `init_distribution.png`
  - `final_population.csv`
  - `final_objectives.csv`
  - `pareto_front.png`

## 5. 验证方案
1. 单元/烟雾测试：
   - `deap_nsga2` 运行后产物文件存在；
   - 结果维度与边界合法。
2. 回归测试：
   - 现有 `random_search` 流程不受影响；
   - `pytest` 全通过。
3. 运行验证：
   - `run_baseline.py --engine deap_nsga2` 可成功产出结果。

## 6. 风险与回滚
风险：
- DEAP `creator` 全局注册重复报错；
- 种群选择阶段在小种群时行为异常。

缓解：
- 使用“已存在即复用”的 creator 注册策略；
- 种群大小默认保持偶数，必要时在代码中容错处理。

回滚：
- 保留 `random_search` 引擎，出现异常时可切回。

## 7. Implementation Plan
- [x] I1: 在 `runner.py` 新增 `deap_nsga2` 实现并接入 `run()` 分发
- [x] I2: 移除/下线 `geatpy_nsga2` 分支与相关测试
- [x] I3: 更新 `run_baseline.py` 引擎枚举
- [x] I4: 更新/新增测试覆盖 `deap_nsga2`
- [x] I5: 执行 `pytest` 与 baseline 实跑

## 8. 关联提交与结果
- Commit: 待补充
- Artifacts: `artifacts/baseline_deap_nsga2_20260225_202542/`
