# Design Docs Ledger（DD）

本目录用于保存方案二项目的“设计迁移记录”（类似 db migrations）。

## 编号规则
- 文件名：`DD-XXXX-<slug>.md`
- `XXXX` 为四位递增编号（0001 开始）
- 一个重大改动至少对应一个 DD 文档

## 状态约定
- `Draft`：草案
- `Approved`：评审通过，可实施
- `Implemented`：代码已落地
- `Superseded by DD-XXXX`：被后续方案替代

## 最小模板
每个 DD 至少包含：
1. 背景与目标
2. 问题定义与非目标
3. 方案对比与取舍
4. 设计细节（接口/流程/数据）
5. 验证与评估计划
6. 风险与回滚
7. Implementation Plan（任务列表）
8. 关联提交/实验结果

## 建议流程
1. Plan Mode 做 research
2. 新建 DD 文档并编号
3. 评审后实现
4. 将 DD 与代码一起提交
