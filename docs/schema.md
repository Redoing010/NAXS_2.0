# 数据与 Schema 契约

本文件收敛 NAXS 3.0 在数据层的结构化约束，覆盖时间序列、事件、因子、知识图谱以及 ClickHouse/Parquet/Qlib 等存储映射。所有 JSON Schema 均位于 `contracts/schemas/`。

## 1. 时间序列与事件 Schema

### 1.1 时间序列 (`timeseries.schema.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "TimeSeriesRow",
  "type": "object",
  "properties": {
    "symbol": {"type": "string"},
    "ts": {"type": "string", "format": "date-time"},
    "freq": {"type": "string", "enum": ["tick","1min","5min","day","week","month"]},
    "fields": {"type": "object", "additionalProperties": {"type": ["number","string","null"]}},
    "source": {"type": "string"},
    "ingested_at": {"type": "string", "format": "date-time"}
  },
  "required": ["symbol","ts","freq","fields"],
  "additionalProperties": false
}
```

### 1.2 事件 (`event.schema.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CorporateEvent",
  "type": "object",
  "properties": {
    "event_id": {"type":"string"},
    "symbol": {"type":"string"},
    "event_type": {"type":"string", "enum":["announcement","earnings","mna","buyback","dividend","management_change","supply_chain"]},
    "event_time": {"type":"string","format":"date-time"},
    "trading_day": {"type":"string","format":"date"},
    "payload": {"type":"object"},
    "source": {"type":"string"},
    "delay": {"type":"integer","minimum":0}
  },
  "required":["event_id","symbol","event_type","event_time"],
  "additionalProperties": false
}
```

## 2. ClickHouse 表结构

### 2.1 日线行情 `ts_1d`

```sql
CREATE TABLE ts_1d
(
  symbol String,
  ts DateTime64(3, 'UTC'),
  open Float64, high Float64, low Float64, close Float64,
  volume Float64, amount Float64,
  source LowCardinality(String),
  ingested_at DateTime64(3, 'UTC')
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (symbol, ts);
```

### 2.2 公司事件 `corporate_event`

```sql
CREATE TABLE corporate_event
(
  event_id String,
  symbol String,
  event_type LowCardinality(String),
  event_time DateTime64(3, 'UTC'),
  trading_day Date,
  payload JSON,
  source LowCardinality(String)
)
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (symbol, event_time);
```

## 3. 数据治理与对齐

- **单位统一**：金额、股价 → CNY；数量 → 万/亿。
- **时区**：存储 UTC，视图层转换 Asia/Shanghai。
- **复权**：前复权字段作为默认回测输入，保留不复权核验。
- **事件对齐**：`event_time → trading_day`，延迟 `delay = N`。
- **质量校验**：缺失阈值 < 1%、跳变 > 8σ、重复去重、日线延迟 < 10 分钟。

### 3.1 Great Expectations 规则片段

```yaml
expectations:
  - expect_table_row_count_to_be_between: {min_value: 1}
  - expect_column_values_to_not_be_null: {column: close, mostly: 0.995}
  - expect_column_values_to_be_between: {column: volume, min_value: 0}
  - expect_compound_columns_to_be_unique: {column_list: [symbol, ts]}
  - expect_column_value_z_scores_to_be_less_than:
      column: close
      threshold: 8
```

## 4. 因子与特征

### 4.1 因子注册 (`factor.schema.json`)

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 因子唯一标识 |
| `name` | string | 展示名称 |
| `owner` | string | 维护团队 |
| `category` | string | 价格/财务/情绪/事件 |
| `source` | string | 数据来源表 |
| `expression` | string | 计算表达式或 SQL/DSL |
| `cleaning` | object | winsorize/normalize/neutralize 配置 |
| `lag` | string | 滞后控制（如 `1d`）|
| `window` | string | 滚动窗口范围 |
| `alignment` | object | 事件对齐策略 |
| `tags` | array | 检索标签 |

#### 示例（`modules/factors/registry.yaml`）

```yaml
factors:
  - id: px_turnover
    name: Turnover Ratio
    owner: quant-core
    category: price_volume
    source: ts_1d
    expression: "Volume / FreeFloatShares"
    cleaning:
      winsorize: {method: mad, k: 5}
      normalize: zscore
      neutralize: {by: [industry_sw, mktcap_log]}
    lag: 1d
    window: "20d:120d"
    tags: [liquidity, style]
  - id: senti_news
    name: News Sentiment
    category: sentiment
    source: news_agg
    expression: "EMA(sentiment_score, 5)"
    alignment: {by: event_time -> trading_day, delay: 1}
    tags: [event, sentiment]
```

### 4.2 特征流水线

`FeaturePipeline.materialize()` 步骤：

1. 加载原始数据（行情/事件/财务）。
2. 根据因子定义执行对齐（事件 → 交易日、滞后/冻结窗口）。
3. 应用清洗策略（去极值、标准化、中性化）。
4. 写入特征存储（Parquet/Arrow/ClickHouse 特征表）。

## 5. Qlib 配置

- **实验参数**：市场 `csi300`、基准 `SH000300`、模型 `LGBModel`、因子列表 `[px_turnover, senti_news, quality_roa, value_bp]`。
- **标签**：`Ref($close, -5)/Ref($close, -1)-1`（未来 5 日收益）。
- **回测策略**：`TopKDropoutStrategy`，`topk=50`，`n_drop=5`，`cost=0.001`，调仓频率 `weekly`。

## 6. 知识图谱 Schema

### 6.1 实体 (`kg-entity.schema.json`)

- `entity_type` 枚举：Company、Person、Product、Event、Indicator。
- `attributes` 存储行业、市值、角色等元数据。

### 6.2 关系 (`kg-edge.schema.json`)

- `relation` 枚举：HOLDING、SUPPLY、COMPETE、ANNOUNCE、LEADS、MENTIONS。
- 支持权重、有效期窗口、附加属性。

### 6.3 Neo4j 索引与查询

```cypher
CREATE INDEX company_code IF NOT EXISTS FOR (c:Company) ON (c.code);
CREATE INDEX event_id IF NOT EXISTS FOR (e:Event) ON (e.event_id);

MATCH (a:Company {code:$code})-[:SUPPLY*1..2]->(b:Company)
OPTIONAL MATCH (a)-[:ANNOUNCE]->(e:Event)
WHERE e.time >= datetime() - duration('P30D')
RETURN a,b,collect(e) AS events;
```

## 7. 报告输出 (`report.schema.json`)

- `metrics`：年化收益、夏普、最大回撤必填，支持 IC、IC_IR。
- `sections`：标题 + Markdown 正文。
- `artifacts`：图表/附件 URI。
- `regime`：当前市场状态与置信度。
- `holdings`：Top 持仓及权重。
- `backtest`：标准化结果（`metrics`、`slices`、`charts`、`explain`）。

## 8. SSE/WebSocket 指令

```json
{"op":"navigate","route":"analysis/backtest","params":{"task_id":"bt_20250920_001"}}
{"op":"patch","target":"chart#equity_curve","payload":{"series":[["2025-01-01",1.0],["2025-01-02",1.01]]}}
{"op":"modal","target":"param-fill","payload":{"missing":["horizon","risk"]}}
{"op":"patch","target":"panel#explain","payload":{"shap_topk":["mom_5","quality_roa"],"events":[...]}}
```

## 9. 数据产品映射

| 存储 | 用途 | 工具 |
| --- | --- | --- |
| Parquet (`data/parquet/`) | 原始与清洗后的时间序列 | Pandas/Polars, Spark |
| ClickHouse (`ts_1d`, `corporate_event`) | 高频查询、特征对齐 | SQL, ch-bench |
| Qlib (`data/qlib_bundle/`) | 模型训练、回测 | Qlib Python API |
| Neo4j | 知识图谱查询 | Cypher, GraphQL |

## 10. 质量与审计字段

- `source`: 原始数据源（akshare/wind/news_crawler 等）。
- `ingested_at`: 入库时间。
- `visibility_ts`: 对外可见时间（DQ 驱动）。
- `quality_score`、`quality_flags`: 用于 DQ 报告。
- `trace_id`: 全链路追踪标记。

> 数据契约须在变更前提 PR，Schema 变动需更新 `version` 与兼容策略，并同步告知下游（Orchestrator、前端、分析脚本）。
