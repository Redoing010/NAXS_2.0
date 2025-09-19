# RUNBOOK — NAXS 3.0 运行手册

本手册覆盖日常运维、紧急故障处理与关键流水线操作，默认部署环境为 Docker Compose 或 Kubernetes。

## 1. 日常任务

| 任务 | 描述 | 频率 | 指令 |
| --- | --- | --- | --- |
| 数据拉取 | 通过 `ops/pull_prices.py` 从 AKShare 抓取 CSI300 日线 | 交易日收盘后 | `python ops/pull_prices.py --universe csi300 --freq day` |
| Qlib Bundle | 构建/更新 Qlib bundle | 每日 | `python ops/build_qlib_bundle.py --start 2016-01-01 --end <today>` |
| 数据质量报告 | 运行 Great Expectations 并生成报告 | 每日 | `python ops/dq_report.py --output reports/dq/$(date +%F)` |
| 回测任务 | 通过 Orchestrator 队列触发回测 | 按需 | `curl -X POST /api/backtests` |
| 报告归档 | 报告上传至 OSS/S3 | 每周 | `python ops/sync_oss.py --src reports --dest s3://naxs-reports/` |

## 2. 数据管道 DAG

### 2.1 `dag_data_pipeline.py`

1. **pull_prices**：调用 `ops/pull_prices.py`，写入 Parquet。
2. **dq_check**：执行 Great Expectations，触发 `dq.alert`。
3. **parquet_to_clickhouse**：将最新分区写入 ClickHouse。
4. **qlib_bundle_update**：刷新 Qlib 数据。

### 2.2 `dag_daily_backtest.py`

1. **materialize_features**：触发 `FeaturePipeline`。
2. **submit_backtest**：调用 `/api/backtests`。
3. **publish_report**：等待 `backtest.completed`，调用 Report Service 生成报告。

## 3. 故障排查

### 3.1 数据延迟

- **症状**：SLO 告警提示日线延迟 > 10 分钟。
- **排查步骤**：
  1. 检查 `ops/pull_prices.py` 日志（`logs/data/*.log`）。
  2. 查看 ClickHouse `system.merges` 是否堆积。
  3. 确认网络/数据源可用性。
  4. 触发手动补数：`python ops/pull_prices.py --replay --start <date>`。

### 3.2 回测卡住

- **症状**：`backtest.scheduled` 无 `backtest.completed`。
- **排查步骤**：
  1. 查看 Orchestrator 日志，检查任务状态机。
  2. 检查 Redis Streams/RabbitMQ backlog。
  3. 登录 Qlib 容器查看 `apps/qlib-adapter/logs/*.log`。
  4. 若任务超时 > 30 分钟，调用 `/api/commands/execute` 重新提交，并将原任务标记 `failed`。

### 3.3 报告生成失败

- **症状**：Report Service 返回 500 或未发布 `report.ready`。
- **排查步骤**：
  1. 检查模板渲染日志（`apps/report-service/logs`）。
  2. 确认 LLM 服务可用（OpenAI proxy/本地模型）。
  3. 手动执行 `python apps/report-service/cli.py render --task <id>` 验证。

## 4. 紧急预案

| 场景 | 处置 |
| --- | --- |
| ClickHouse 宕机 | 切换到 Parquet 直读模式，API 降级返回缓存数据；通知 DBA 恢复。|
| Redis/RabbitMQ 故障 | Orchestrator 启用内存队列缓冲，限制 `run_backtest` 并发，优先处理已接单任务。|
| LLM 额度耗尽 | 降级为模板+历史指标，禁用长文本总结，并在 UI 弹出提示。|
| 数据源限流 | 切换备用数据源（Wind）、降低拉取频率，扩容缓存命中率。|

## 5. 关键指标与告警阈值

详见 `docs/SLO_SLA_SLI.md`。建议在 Grafana 设置以下告警：

- `data_ingest_latency > 10m` 持续 2 个周期。
- `backtest_success_rate < 0.99` 任意 1 小时窗口。
- `orchestrator_p95_latency > 2.5s` 持续 5 分钟。
- `gateway_http_5xx_rate > 1%` 持续 5 分钟。

## 6. 常用命令

```bash
# 启动本地服务
docker compose -f infra/docker/docker-compose.yml up -d

# 运行单测
npm test --workspace apps/api-gateway
pytest -q apps/qlib-adapter

# 导入 ClickHouse Schema
clickhouse-client --queries-file docs/sql/clickhouse_schema.sql

# 触发一次回测（示例）
curl -X POST http://localhost:8080/api/backtests \
  -H 'X-API-Key: demo' \
  -H 'X-Trace-Id: manual-$(uuidgen)' \
  -d '{
    "universe": "csi300",
    "features": ["px_turnover", "senti_news"],
    "label": "Ref($close, -5)/Ref($close, -1)-1",
    "start": "2023-01-01",
    "end": "2023-12-31",
    "rebalance": "weekly"
  }'
```

## 7. 升级流程

1. Schema 变更：更新 `contracts/schemas`、`docs/schema.md`，通过数据回放验证。
2. 服务发布：合并主分支后触发 `cd.yml`，镜像推送至镜像仓库。
3. 数据迁移：提前执行 ClickHouse/Neo4j DDL，并在变更窗口确认指标稳定。
4. 回滚策略：保留上一版本镜像，CI 自动生成 `rollback` 清单。

## 8. 联系人与职责

| 角色 | 团队 | 职责 |
| --- | --- | --- |
| Data Lead | Data Platform | 数据接入、质量治理 |
| Quant Lead | Quant Core | 因子、回测、策略治理 |
| Agent Lead | AI Platform | Orchestrator 与工具链 |
| Frontend Lead | Experience | LLM UI、SSE 协议 |
| SRE | Infra | CI/CD、监控、告警 |

> 如遇未覆盖场景，记录在 `docs/RUNBOOK.md` 并通过 PR 更新，确保知识库持续演进。
