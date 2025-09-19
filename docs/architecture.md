# NAXS 3.0 架构蓝图与交付基线

> 本文是 NAXS 3.0 的起始工程文档，覆盖目录规范、接口契约、数据与因子 schema、Agent 协作协议、流水线脚本框架、监控指标、CI/CD、验收标准与里程碑。

## 0. 仓库骨架（Monorepo 多应用 + 多语言）

```text
repo/
  contracts/
    openapi.yaml                 # 统一 REST 契约（3.1）
    asyncapi.yaml                # 事件/流式契约（2.6）
    schemas/                     # JSON-Schema / Avro / Protobuf
      factor.schema.json
      kg-entity.schema.json
      kg-edge.schema.json
      backtest-request.schema.json
      report.schema.json
      timeseries.schema.json
      event.schema.json
      command.schema.json
  apps/
    api-gateway/                 # Node/TS（Fastify）+ Zod 校验 + RBAC
    orchestrator/                # Agent Orchestrator（Planner/Executor/Critic）
    qlib-adapter/                # Python：Qlib 训练/回测/特征存储适配
    data-service/                # Python：数据中台 API（读取 Parquet/ClickHouse）
    kg-service/                  # Neo4j/TigerGraph GraphQL + 自定义算法
    report-service/              # 报告生成（LLM+模板）
    web/                         # React + Tailwind + shadcn/ui（LLM 入口 + 二级界面）
  modules/
    data/                        # 数据接入/清洗/对齐/质量检测
      interfaces.py
      akshare_source.py
      wind_source.py (stub)
      news_crawler/
      calendar.py
      parquet_store.py
      dq.py
      qlib_writer.py
      utils.py
    factors/                     # 因子库（注册中心 + 清洗模板）
      registry.yaml
      base/price_volume/
      base/financial/
      sentiment/
      event_driven/
      pipelines/
        feature_pipeline.py
        cleaning.py
        alignment.py
    agents/                      # 子 Agent 实现（工具层封装）
      retriever/
      analyzer/
      reporter/
      preference/
      tools/
        qlib_tool.py
        kg_tool.py
        data_tool.py
        chart_tool.py
  data/
    parquet/                     # 分区: dt=YYYY-MM-DD/symbol=xxxx
    qlib_bundle/                 # Qlib 输出
  ops/
    pull_prices.py
    build_qlib_bundle.py
    dq_report.py
    sync_oss.py
    airflow_dags/
      dag_data_pipeline.py
      dag_daily_backtest.py
  configs/
    app.yaml
    data.yaml
    logging.yaml
    feature_store.yaml
  infra/
    docker/
      docker-compose.yml
      Dockerfile.api
      Dockerfile.qlib
      Dockerfile.web
    k8s/
      gateway-deploy.yaml
      orchestrator-deploy.yaml
      redis.yaml
      clickhouse.yaml
      neo4j.yaml
  .github/
    workflows/
      ci.yml
      cd.yml
  docs/
    schema.md
    data-dictionary.md
    RUNBOOK.md
    SLO_SLA_SLI.md
    SECURITY.md
    architecture.md
```

## 1. 总体架构图

```mermaid
flowchart LR
  subgraph Frontend[交互前端]
    A[LLM 对话入口\n二级操作界面] --> B[可视化组件\n指数/选股/解释]
  end

  subgraph Gateway[API Gateway]
    G1[Auth + RBAC + RateLimit] --> G2[REST/gRPC]
  end

  subgraph Orchestrator[Agent Orchestrator]
    O1[意图识别/任务分解] --> O2[Planner]
    O2 --> O3[Executor]
    O3 --> O4[Critic/Verifier]
  end

  subgraph Services[领域服务]
    D[Data Service\n(数据中台)]
    Q[Qlib Adapter\n(训练/回测)]
    K[KG Service\n(Neo4j/TG)]
    R[Report Service\n(模板+LLM)]
  end

  subgraph Infra[基础设施]
    MQ[(Redis Streams/RabbitMQ)]
    CK[(ClickHouse/Parquet)]
    NEO[(Neo4j/TigerGraph)]
    S3[(OSS/S3)]
    MON[监控与日志\n(Prom/Grafana/ELK)]
  end

  A -->|SSE/WebSocket| G1
  G2 --> O1
  O3 <--> MQ
  O3 -->|tool.call| D
  O3 -->|tool.call| Q
  O3 -->|tool.call| K
  O3 -->|tool.call| R
  D <--> CK
  Q <--> CK
  R --> S3
  K <--> NEO
  O3 --> MON
  G1 --> MON
```

## 2. 数据接入与中台

- **Schema**：参见 `docs/schema.md` 以及 `contracts/schemas/*.json`。
- **治理原则**：单位统一为 CNY、UTC 存储、前复权基准、事件对齐 `event_time → trading_day`、延迟 SLA 日线 < 10 分钟。
- **ClickHouse**：`ts_1d` 与 `corporate_event` 表结构用于基础行情与公司事件存储。
- **质量**：Great Expectations 规则保障缺失率、跳变、重复与异常延迟。

## 3. 因子工程与 Qlib 适配

- 因子注册中心 `modules/factors/registry.yaml` 定义表达式、清洗、对齐、滞后与标签。
- `FeaturePipeline` 负责原始数据加载、对齐、清洗、写入特征存储。
- Qlib 实验通过 `apps/qlib-adapter` 管理配置，支持 LGBM 模型与 TopK 策略回测。
- 回测评估标准化输出年化收益、夏普、最大回撤、IC、因子贡献等字段，供报告与前端复用。

## 4. Agent 架构与工具调用

- 子 Agent：Retriever（数据检索）、Analyzer（回测/归因）、Reporter（模板+LLM）、Preference（画像与策略重排）。
- Orchestrator 以 Function Calling 模式调度工具，统一 JSON Schema (`command.schema.json`)。
- 工作流：UI → Gateway → Orchestrator → Retriever/Analyzer/Reporter → SSE 推送结果。

## 5. 市场状态识别与动态权重

- Regime 模型：HMM/GMM/LSTM/Transformer，周频输入波动率、换手率、行业轮动、宏观因子。
- 权重策略：`weight_policy.yaml` 定义 `bayesian_ucb` 参数、冷却期与各 Regime 映射。

## 6. 研报生成

- 报告模板使用 Jinja2，结合指标、贡献、风险与持仓表格。
- 报告 schema 由 `report.schema.json` 限定，生成 Markdown/HTML/PDF。

## 7. API 契约

- REST 契约：`contracts/openapi.yaml` 覆盖 `/intents`、`/commands/execute`、`/backtests`、`/features`、`/reports/{id}`。
- 关键组件包括 `Trace-Id`、`Idempotency-Key`、错误响应统一结构。

## 8. 事件与流式接口

- AsyncAPI (`contracts/asyncapi.yaml`) 订阅 `data.ingested`、发布 `backtest.completed`、`report.ready` 等事件。
- 消息载荷依赖 JSON Schema（如 `timeseries.schema.json`、`report.schema.json`）。

## 9. 知识图谱

- 实体：Company、Person、Product、Event、Indicator。
- 关系：HOLDING、SUPPLY、COMPETE、ANNOUNCE。
- Neo4j 索引：`company_code`、`event_id`；典型查询示例见 `docs/schema.md`。

## 10. 实时系统与工程保障

- 指标：数据延迟、回测成功率、Orchestrator P95 延迟、API 可用性。
- 日志：结构化 JSON，Trace-Id 贯穿；OpenTelemetry → Jaeger/Tempo。
- 权限：RBAC（viewer/pro/admin/partner）与审计追踪。

## 11. CI/CD 与环境

- GitHub Actions（`ci.yml`）安装 Node 20 与 Python 3.11，执行单测与 Docker 构建。
- Docker Compose 启动 gateway、orchestrator、redis、clickhouse、neo4j。

## 12. 前端协议与组件

- SSE/WebSocket 指令：`navigate`、`patch`、`modal`、`panel`。
- UI 组件：指数 TickerBar、策略卡片、选股表格、解释面板、报告 Viewer。

## 13. WBS 阶段规划

### 阶段一（数据与基础设施，3–4 周）

| 编号 | 工作项 | 产出 | 验收标准 |
| --- | --- | --- | --- |
| B1 | 数据模式与字典 | `docs/schema.md`, `data-dictionary.md` | 字段/单位/时区全覆盖；GE 校验通过 |
| B2 | 接入 AKShare/交易日历 | `modules/data/akshare_source.py`, `calendar.py` | 2016–2025 CSI300 拉取成功，缺失 <1% |
| B3 | Parquet 存储/增量 | `parquet_store.py` | 分区读写、天级增量 < 2 分钟 |
| B4 | DQ 报表 | `ops/dq_report.py` + `/reports/dq/` | 生成 Markdown/CSV 报告 |
| B5 | Qlib 适配 | `qlib_writer.py` + `apps/qlib-adapter` | 训练/回测可跑，导出指标 JSON |
| B6 | KG 初版 | `apps/kg-service` | 公司/公告入图与查询 |

### 阶段二（智能体与交互，3 周）

| 编号 | 工作项 | 产出 | 验收标准 |
| --- | --- | --- | --- |
| A1 | 意图解析 API | `/intents` | 6 类意图 F1 > 0.8 |
| A2 | 工具封装 | `agents/tools/*` | 回测/KG/数据 function calling |
| A3 | Orchestrator | `apps/orchestrator` | E2E 输入→报告（SSE 推流） |
| A4 | 前端二级界面 | `apps/web` | 三大指数+回测页+解释页可用 |

### 阶段三（动态策略与解释性，3 周）

| 编号 | 工作项 | 产出 | 验收标准 |
| --- | --- | --- | --- |
| S1 | Regime 训练 | `models/regime/*` | Regime 稳定（NMI > 0.6） |
| S2 | 动态权重 | `weight_policy.yaml` + 实现 | 年化提升、回撤下降 |
| S3 | 解释性 | SHAP/Permutation | 因子 Top-K 稳定；报告展示 |

## 14. 验收指标（MVP）

- 功能：输入“详细分析我持有的 000831”，3s 内返回指数栏、个股概览、回测、解释、事件时间线。
- 质量：IC > 0、回测年化高于基准 5%+、最大回撤低于基准。
- 性能：首屏 < 2.5s、回测 < 30s（周频/TopK50）。
- 运维：90%+ 路径可追踪，失败自动重试/告警。

## 15. 风险与对策

- 数据合规：签约与权限隔离、敏感字段脱敏、审计追踪。
- 模型漂移：每周评估、阈值预警、自动回滚。
- 耦合上升：契约先行，工具层隔离实现。
- 双写不一致：单写多读，回测与生产共享镜像与配置。

## 16. 快速起步（本周行动）

1. 创建 `contracts/` 契约文件与 docs 起始内容。
2. 完成 B1/B2/B3 最小可用版本（日线/CSI300）。
3. 启动 ClickHouse 表并运行 `dq_report` 生成首份报告。
4. 打通 Qlib 最小回测，产出 `BacktestResult.json`。
5. Web 端接入 SSE，渲染指数栏与收益曲线。

## 17. Codex 服务化要点

- 意图体系（`SectorShortTermPick`、`ThematicPick`、`PositionSizing`、`ExplainFactor`、`BacktestQuick`、`EventReplay`）。
- 指令契约统一 `command.schema.json`，核心命令 `screen_sectors`、`run_backtest`、`position_size`、`kg_timeline` 等。
- API 分层：`/api/intents/parse`、`/api/commands/execute`、`/api/screen/sectors` 等。
- 前端协议扩展：`modal#param-fill`、`panel#explain`。

## 18. Analyzer 输出

- Position Sizer：保守/基准/进取三档位，返回仓位与风险增量。
- Backtest：统一 `metrics`、`slices`、`charts` 结构，支持缓存降级。

## 19. 语义词典

- 板块、主题、周期、风险词典用于参数抽取，详见 `docs/data-dictionary.md`。

## 20. E2E 用例

- E2E-01：SectorShortTermPick → `screen_sectors` → `run_backtest`。
- E2E-02：PositionSizing → `position_size`。
- E2E-03：BacktestQuick → `run_backtest`。
- 测试脚本模板：`tests/test_e2e_codex.py`，校验 trace_id 与 SLA。

## 21. 渐进式落地路线

- Sprint 1：上线 `/api/intents/parse` 与 `/api/commands/execute` 骨架，完成 SectorShortTermPick 与 PositionSizing 模拟链路；前端接 `modal` 与 `panel`。
- Sprint 2：回测输出统一化、缓存降级、KG 时间线与解释面板。
- Sprint 3：Regime + 权重展示、A/B 离线评估、因子注册中心 UI。

## 22. 安全与合规（研究域）

- 只读执行（拒绝下单/资金指令）、资源保护（回测队列/并发限制）、审计（话术、意图、参数、模型版本、数据快照）。

## 23. 参考附件

- 契约：`contracts/openapi.yaml`, `contracts/asyncapi.yaml`。
- Schema：`contracts/schemas/*.json`。
- 运行维护：`docs/RUNBOOK.md`, `docs/SLO_SLA_SLI.md`, `docs/SECURITY.md`。
