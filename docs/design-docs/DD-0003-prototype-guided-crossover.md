# DD-0003: 原型引导交叉（P3）设计与落地

- Status: Implemented
- Date: 2026-02-25
- Author: 杨振国（草案由 Copilot 协助整理）
- Scope: `scheml_2` P3（对应 DD-0001）

## 1. 背景与目标
P2 已完成 DEAP NSGA-II 主循环替换，当前交叉仍为标准 SBX，缺少“向高质量施肥模式靠拢”的知识引导能力。P3 的目标是在不破坏现有接口的前提下，加入“原型引导交叉”。

## 2. 问题定义与非目标
### 2.1 问题定义
实现如下流程：
1. 从当前种群中提取精英个体；
2. 基于精英个体构建施肥原型集合；
3. 交叉时对子代施加原型牵引，且牵引强度随代数退火。

### 2.2 非目标
- 本次不引入协同/拮抗耦合变异（P4）；
- 本次不引入动态精英保留（P5）；
- 本次不依赖外部聚类库（采用轻量 numpy KMeans）。

## 3. 方案与取舍
### 方案 A：仅使用 SBX
实现简单，但无法引入高产结构先验。

### 方案 B：SBX + 原型牵引（选择）
先进行父代融合，再按最近原型牵引：
- 保留进化算法的探索性；
- 强化向高质量模式收敛；
- 可通过退火系数控制早期引导、后期自由探索。

## 4. 设计细节
### 4.1 新增算子模块
新增 `src/fertopt/operators/crossover.py`：
- `build_elite_prototypes(...)`：从精英样本构建原型
- `prototype_guided_crossover(...)`：执行原型引导交叉

### 4.2 原型构建
- 精英选择：按多目标加和分数从小到大选取前 `elite_ratio`
- 聚类方法：轻量 KMeans（固定迭代）
- 原型数：`prototype_count`

### 4.3 交叉公式
对阶段 `s` 与养分 `j`：
1. 父代凸组合：
   $$x_s^{(c)}(j)=\alpha_s x_s^{(A)}(j)+(1-\alpha_s)x_s^{(B)}(j),\ \alpha_s\sim Beta(2,2)$$
2. 原型牵引：
   $$x_s^{(c)}(j)\leftarrow (1-\gamma_s)x_s^{(c)}(j)+\gamma_s a_s^{(k^*)}(j)$$
3. 退火：
   $$\gamma_s=\gamma_0\cdot(1-\frac{g}{G_{max}})$$
4. 边界投影：
   $$x\leftarrow clip(x,lb,ub)$$

## 5. 配置扩展
在 `crossover` 配置中新增：
- `prototype_count`：原型数量
- `elite_ratio`：精英比例
- `kmeans_iters`：KMeans 迭代次数

## 6. 验证方案
1. 算子单元测试：
   - 输出维度不变；
   - 不越界；
   - 与最近原型距离有收敛倾向。
2. 集成测试：
   - `deap_nsga2` 全流程可运行；
   - 产物文件完整。

## 7. 风险与回滚
风险：
- 牵引过强导致种群多样性下降；
- 原型数量过多导致噪声引导。

缓解：
- 默认 `gamma0` 中等强度并随代数退火；
- 对 `prototype_count` 与 `elite_ratio` 做合理上下限裁剪。

回滚：
- 保留标准 SBX fallback 路径；
- 如出现异常可暂时关闭原型牵引（`gamma0=0`）。

## 8. Implementation Plan
- [x] I1: 新增 `operators/crossover.py`
- [x] I2: 扩展 `CrossoverConfig`
- [x] I3: 在 `runner.py` 的 DEAP 循环接入原型引导交叉
- [x] I4: 新增单元测试
- [x] I5: 执行 `pytest` 与 baseline 验证

## 9. 关联提交与结果
- Commit: 待补充
- Artifacts: `artifacts/baseline_deap_nsga2_20260225_210019/`
