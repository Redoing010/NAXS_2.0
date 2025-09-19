# SLI / SLO / SLA 指标

该文档定义 NAXS 3.0 在数据、计算与交互三个维度的可观测性指标及目标。

## 1. 数据层

| 指标 | 类型 | 定义 | 目标 (SLO) | 告警阈值 |
| --- | --- | --- | --- | --- |
| 数据延迟（日线） | SLI | `now() - latest(ts)` | < 10 分钟 | > 10 分钟持续 2 个周期 |
| 数据延迟（分钟） | SLI | `now() - latest(ts)` | < 30 秒 | > 60 秒持续 5 分钟 |
| 缺失率 | SLI | `missing_rows / total_rows` | < 1% | > 1% 任意窗口 |
| DQ 通过率 | SLI | `passed_expectations / total` | ≥ 98% | < 95% 持续 1 小时 |
| DQ 告警恢复时间 | SLA | 告警到恢复 | < 30 分钟 | > 30 分钟升级 |

## 2. 回测与模型

| 指标 | 类型 | 定义 | 目标 | 告警 |
| --- | --- | --- | --- | --- |
| 回测成功率 | SLI | `successful_tasks / total_tasks` | ≥ 99% | < 99% 任意 1 小时 |
| 回测完成时长 P95 | SLI | `duration_p95` | < 30 秒 | > 45 秒 |
| Orchestrator 工单延迟 P95 | SLI | 接口响应 P95 | < 2.5 秒 | > 3 秒 |
| 模型漂移评分 | SLI | `abs(recent_ic - baseline_ic)` | < 0.05 | > 0.1 |
| 回测 SLA | SLA | 单次回测完成 | ≤ 5 分钟 | 超时告警 + 自动重试 |

## 3. API 与前端

| 指标 | 类型 | 定义 | 目标 | 告警 |
| --- | --- | --- | --- | --- |
| API 可用性 | SLO | `1 - 5xx/total` | ≥ 99.99% | 5xx > 1% |
| API 延迟 P95 | SLI | `/api` 请求 P95 | < 300 ms | > 500 ms |
| SSE 首帧延迟 | SLI | `time_to_first_chunk` | < 2.5 s | > 3 s |
| SSE 中断率 | SLI | `dropped_connections / total` | < 0.5% | > 1% |

## 4. 监控与告警渠道

- **Prometheus**：采集服务指标，Grafana 展示仪表盘。
- **Tempo/Jaeger**：Trace 采样率 ≥ 10%，Orchestrator 关键路径强制记录。
- **Loki/ELK**：结构化 JSON 日志，支持 traceId 搜索。
- **告警**：通过 Alertmanager → 飞书/Slack，关键指标支持短信兜底。

## 5. 采样与标签

- 所有指标带标签：`service`, `env`, `region`, `intent`, `command`。
- 数据延迟与回测指标需按 `universe`、`freq` 细分。
- SSE 指标按 `route`、`component` 细分。

## 6. 评审与迭代

- 每周 SLI/SLO 复盘，评估 SLA 偏差。
- 新增服务需在上线前注册 SLI 与告警策略。
- 失败指标需在 24 小时内制定整改计划。

> 指标定义变更需要更新本文件、Grafana Dashboard 与 Alertmanager 规则，并向 Oncall 团队公告。
