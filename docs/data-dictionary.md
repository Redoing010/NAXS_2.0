# 数据字典与语义映射

该文档描述核心数据集、字段含义、单位约束与 Codex 语义词典，用于支撑意图解析、因子计算与报告生成。

## 1. 时间序列（行情）

| 字段 | 中文名 | 类型 | 单位/格式 | 说明 |
| --- | --- | --- | --- | --- |
| `symbol` | 证券代码 | string | `XXXXXX.XX` | 主键之一，统一 6 位代码 + 交易所后缀 |
| `ts` | 时间戳 | datetime | ISO8601, UTC | 精确到毫秒的采集时间 |
| `freq` | 频率 | enum | tick/1min/5min/day/week/month | 驱动聚合粒度 |
| `open` | 开盘价 | float64 | CNY | 存储于 `fields.open` |
| `high` | 最高价 | float64 | CNY | `fields.high` |
| `low` | 最低价 | float64 | CNY | `fields.low` |
| `close` | 收盘价 | float64 | CNY | `fields.close` |
| `volume` | 成交量 | float64 | 股 | `fields.volume`，缺失补 0 |
| `amount` | 成交额 | float64 | CNY | `fields.amount` |
| `adj_factor` | 复权因子 | float64 | - | 可选，存在于特征表 |
| `source` | 数据源 | string | - | akshare/wind 等 |
| `ingested_at` | 入库时间 | datetime | UTC | ETL 入库时间 |

> 业务规则：`high >= max(open, close)`、`low <= min(open, close)`、`volume >= 0`。

## 2. 事件数据

| 字段 | 中文名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `event_id` | 事件 ID | string | 主键，UUID/哈希 |
| `symbol` | 证券代码 | string | 可为空（如宏观事件）|
| `event_type` | 事件类型 | enum | announcement/earnings/mna/buyback/dividend/management_change/supply_chain |
| `event_time` | 事件时间 | datetime | UTC |
| `trading_day` | 对齐交易日 | date | Asia/Shanghai（字符串）|
| `payload` | 事件 payload | object | JSON，自由扩展 |
| `source` | 数据源 | string | 公告/新闻等 |
| `delay` | 披露延迟 | integer | T+N 天 |

## 3. 因子元数据

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 因子 ID，建议 `snake_case` |
| `category` | enum | price_volume/financial/sentiment/event_driven/custom |
| `expression` | string | SQL/DSL/Python 表达式 |
| `lag` | string | 例如 `1d`、`5m` |
| `window` | string | `start:end` 格式，如 `20d:120d` |
| `cleaning.winsorize` | object | { method: mad, k:5 } |
| `cleaning.normalize` | enum | zscore/minmax/quantile |
| `cleaning.neutralize.by` | array | [industry_sw, mktcap_log] |
| `alignment.by` | string | 事件对齐映射描述，如 `event_time -> trading_day` |
| `alignment.delay` | integer | 滞后天数 |

## 4. 特征存储字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `dt` | date | 交易日 |
| `symbol` | string | 证券代码 |
| `values` | object | 键值对，`{factor_id: value}` |
| `quality_flags` | string | 异常标记（outlier/missing）|
| `visibility_ts` | datetime | 数据可见时间 |

## 5. ClickHouse/Parquet 目录约定

| 路径 | 描述 |
| --- | --- |
| `data/parquet/dt=YYYY-MM-DD/symbol=XXXXXX` | 日线分区存储 |
| `data/parquet/freq=1min/dt=YYYY-MM-DD/` | 分钟级分区 |
| `data/qlib_bundle/` | Qlib bundle 输出 |
| `reports/dq/` | 数据质量报告 |

## 6. 语义词典（意图抽取）

```yaml
sectors:
  新能源: [光伏, 储能, 锂电, 充电桩, 动力电池]
  TMT: [AI, 服务器, GPU, 大模型, 算力]

horizon:
  短线: {label: "5-10D"}
  波段: {label: "1-3M"}
  中长线: {label: "6-12M"}

risk_words:
  激进: {risk_level: high}
  稳健: {risk_level: medium}
  保守: {risk_level: low}
```

## 7. Regime 模型输入

| 指标 | 描述 |
| --- | --- |
| `hv_20` | 20 日历史波动率 |
| `iv_proxy` | 隐含波动率替代（ETF 期权）|
| `turnover_ratio` | 市场换手率 |
| `industry_rotation` | 行业轮动指标 |
| `macro_pmi` | PMI 数据 |
| `macro_cpi` | CPI 数据 |
| `macro_shrz` | 社融总量 |

输出：`regime_id ∈ {bull, range, bear, shock}`、`confidence ∈ [0,1]`。

## 8. 回测指标映射

| 字段 | 描述 | 计算 |
| --- | --- | --- |
| `ann_return` | 年化收益 | `(1+total_return)^(252/trading_days)-1` |
| `sharpe` | 夏普比率 | `excess_return / std_dev` |
| `max_dd` | 最大回撤 | `max(peak - trough)/peak` |
| `ic` | 信息系数 | `corr(pred, future_ret)` |
| `ic_ir` | IC_IR | `mean(IC)/std(IC)` |
| `win_rate` | 胜率 | `wins / trades` |

## 9. 报告结构

| Section | 内容 |
| --- | --- |
| `核心结论` | 年化收益、夏普、回撤、驱动因子、风险点 |
| `因子归因` | SHAP/Permutation 结果、贡献树 |
| `市场状态` | 当前 Regime、权重调整说明 |
| `持仓与交易` | Top 持仓、调仓日志 |

## 10. 审计字段

| 字段 | 说明 |
| --- | --- |
| `trace_id` | 由 Gateway 注入，贯穿全链路 |
| `request_id` | UI 层传入，辅助排障 |
| `model_version` | 当前模型或因子版本 |
| `data_snapshot` | 回测数据快照时间戳 |
| `user_role` | viewer/pro/admin/partner |

> 所有字段变更须在 `docs/schema.md` 与 `contracts/schemas` 同步更新，并通过数据质量作业验证。
