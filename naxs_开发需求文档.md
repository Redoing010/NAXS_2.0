# Naxs 智能投研系统 - 整体开发流程文档

**文档信息**  
- **文档版本**：1.0  
- **文档日期**：2025年9月17日  
- **文档作者**：项目负责人兼产品经理（Grok 协助起草）  
- **文档目的**：提供 Naxs 系统的完整开发流程指导，覆盖所有模块（数据层、特征与因子层、模型与回测层、Agent 层、应用层、基础设施与安全）。基于整体蓝图，采用敏捷方法，确保可迭代开发。流程分为准备、开发、测试、上线和维护阶段。时间线压缩为 MVP 核心（1个月内跑通 Demo 链路），后续扩展到 Beta（额外1-2个月）和商业化（总 12 个月）。  

**假设与约束**：
- **总时间线**：MVP（1个月，2025/09/17 - 2025/10/17）；Beta（至 11 月底）；商业化（至 2026/09）。
- **团队配置**：
  - 数据工程师（2人）
  - 后端开发（3人）
  - AI 工程师（2人）
  - 量化研究员（2人）
  - 前端工程师（2人）
  - UI/UX 设计师（1人）
  - DevOps/安全工程师（1人）
  - 合规专员（1人）
  - 产品经理（1人）
- **技术栈**：
  - 后端：Node.js、Python Flask/Django
  - 数据库：PostgreSQL、S3/Parquet、Redis
  - AI：LangChain/Temporal（Agent）、Qlib（回测）
  - 前端：React + TailwindCSS
  - 基础设施：Docker、Kubernetes、AWS/GCP
- **依赖**：外部数据源（聚宽等）需 API Key；初期开发用 Mock 数据模拟。
- **后续补充项**：API Key、真实数据样本、团队技能评估、预算细节、Qlib 集成文档。

---

## 1. 项目概述
### 1.1 背景与愿景
Naxs 是一个 **AI 驱动的智能投研助手**，整合多源数据、量化因子、回测引擎和 Agent 协作，提供股票研究、策略分析和合规报告。  
**愿景**：成为机构级投研工具，支持自定义策略和实时信号。

### 1.2 整体目标与阶段
- **MVP（1个月）**：跑通核心链路（数据接入 → 因子计算 → 回测 → Agent 报告），产出端到端 Demo。
- **Beta（2个月）**：添加策略切换、仓位建议、自定义因子、实时信号。
- **商业化（12个月）**：API 支持、合规审计、大规模部署。

**成功指标**：
- MVP：全链路延迟 <5s、准确率 >90%
- Beta：用户满意度 >80%
- 商业化：支持 1000+ 用户/日

---

## 2. 开发方法论与工具
- **方法论**：敏捷 Scrum，每 2 周 Sprint；每日 Stand-up、周末回顾。任务跟踪采用 Kanban。
- **工具栈**：
  - 项目管理：Jira/Asana
  - 版本控制：GitHub/GitLab
  - CI/CD：GitHub Actions/Jenkins
  - 文档：Notion/Confluence
  - 测试：Jest (前端)、Pytest (后端)、Selenium (E2E)
  - 监控：Prometheus + Grafana

---

## 3. 开发流程阶段

### 3.1 准备阶段（Week 1: Day 1-7）
**目标**：团队对齐、需求细化、环境搭建。  
**活动**：
- Kick-off 会议：审视框架，明确角色与责任。
- 需求收集：基于用户故事和模块流程图细化需求。
- 环境搭建：Docker Compose 建本地栈、Mock 数据生成。
- 风险评估：如数据源不稳定 → 启用 Mock。

**输出**：需求 Backlog、架构图（UML/ER）、初始仓库。

---

### 3.2 开发阶段（Week 2-3: Day 8-21）
采用模块化并行开发，优先保障 **MVP 核心链路**。每个模块拆解为任务 → 设计 → 开发 → 单测。

#### 3.2.1 数据层（T1）

**目标（Outcome）**：构建稳定、可追溯、可审计的数据接入与治理层，为因子层与 Agent 提供高质量数据。确保 **延迟≤T+1**，质检覆盖≥30条规则，数据契约严格执行。

**用户故事**：As a system, I want clean data ingestion so that factors can be computed accurately.

---

### A) 数据源与采集
- **主数据源**：AkShare（无 Key），覆盖价量、财务、公告、新闻等；
- **备用/扩展源**：Tushare Pro、聚宽、交易所公开文件；
- **采集方式**：
  - Python + httpx/aiohttp 异步拉取；
  - Cron 定时任务（每日收盘后批处理 + 盘中快照）。

**接口示例**：
```python
import akshare as ak
ak.stock_zh_a_hist(symbol="000001", start_date="20210101", end_date="20211231")
```

---

### B) 数据清洗与标准化
- 缺失值：`ffill/bfill` 或丢弃；
- 异常值：3σ/箱线图检测，必要时 winsorize；
- 字段重命名：统一 schema → `symbol, date, open, high, low, close, volume, adj_factor`；
- 时间：全部 UTC；交易日历：`tool_trade_date_hist_sina()`；
- 复权：保存原始与后复权价，口径标记 `price_type`。

---

### C) 存储层设计
- **冷热分离**：
  - 热数据：PostgreSQL（近 1 年，索引 (symbol,date)）；
  - 冷数据：OSS/S3 Parquet（分区：市场/日期/证券）；
- **缓存**：Redis（短期 K 线/财务指标，TTL=24h）。

**表设计**：
- `price_ohlcv(symbol, date, open, high, low, close, volume, adj_factor, visibility_ts, source)`
- `fundamental_quarter(symbol, period, revenue, net_profit, roe, eps, visibility_ts, source)`

---

### D) 数据契约与质检
- 每条记录绑定：`visibility_ts`（可见时间）、`source`、`license`；
- 质检指标：缺失率、异常波动、字段对齐（如市值=股价×股本）、财务平衡表勾稽；
- 违规处理：写入 `qc_flags`，严重错误触发告警事件。

---

### E) 调度与通知
- 流程：`ingest → clean → store → qc → publish → notify`；
- 工具：Temporal/Prefect；
- 事件：
  - `naxs.data.ingested` {source, entity, count, partition, trace_id}
  - `naxs.data.contract_violation` {rule_id, sample, severity, trace_id}

---

### F) API 合同
- `POST /ingest/tasks` {source, entity, range} → {jobId}
- `GET /ingest/jobs/:id` → {status, metrics, trace_id}
- `GET /data/:symbol?start=YYYY-MM-DD&end=YYYY-MM-DD`

**错误码**：400 参数非法；409 重复任务；503 数据源不可用。

---

### G) 验收标准（DoD）
- 数据延迟 ≤ T+1；
- 质检规则 ≥30 条，覆盖价量/财务/交易日历；
- 接口错误率 <0.5%；
- 任意日期/证券可追溯到 `source` 与 `visibility_ts`。

**责任分工**：数据工程师（采集与清洗）、后端开发（API）、合规专员（契约）。

---

#### 3.2.2 特征与因子层（T2） 特征与因子层（T2）

**目标（Outcome）**：在不引入“前视偏差/幸存者偏差”的前提下，构建**可回放、可解释、可扩展**的因子库与特征仓，向上服务选股与回测，向下对接数据契约与口径。MVP 以日频为主，覆盖 A 股 5 年历史，单次全量构建 ≤ 30 分钟，日增量 ≤ 3 分钟。

**用户故事**：As a quant, I want factor scores so that I can rank stocks.

---

### A) 口径与反前视（Data Contract & Anti‑Leakage）
- **可见性时间**：所有特征均携带 `visibility_ts`（UTC），表示“数据在系统中**可被使用**的最早时间”。
- **滞后规则（Lagging）**：日频因子在 **t 日收盘后**才可见，用于 **t+1** 的选股/调仓；`lag=1`。
- **复权口径**：默认后复权（adj_close/adj_factor），同时保存原始收盘价以便核对。
- **停复牌与 ST**：停牌日不计算新值；ST/退市标记入过滤字段。
- **幸存者偏差**：股票名录由交易日日历切片获值；退市/摘牌样本**保留**在历史窗口。

---

### B) 因子目录（MVP）与公式（Formula Spec）
> 统一字段命名：小写下划线；单位与币种标准化（RMB）；行业口径：中证一级行业（`industry_1`）。窗口长度以交易日计。

1. **Value 价值类**  
   - `pe_ttm`: `price_ttm_eps = adj_close / max(eps_ttm, 1e-6)`；若无 `eps_ttm`，回退为 `net_profit_ttm / shares_basic_ttm`。  
   - `pb`: `adj_close / (net_assets / shares_outstanding)`。
2. **Growth 成长类**  
   - `roe_q`: `net_profit_q / equity_q`（季度），拉直至日频并 `ffill`；
   - `eps_g_4q`: `(eps_ttm / eps_ttm_lag_4q) - 1`。
3. **Momentum 动量类**  
   - `mom_20`: `adj_close / adj_close.shift(20) - 1`（去极值后 zscore）；
   - `rsi_14`: 经典 RSI 公式（Wilder），对缺口进行平滑处理。
4. **Funds 资金流/交易特征**  
   - `turnover_20`: 20 日平均换手率；
   - `volatility_20`: `std(log(adj_close).diff(), 20) * sqrt(252)`。
5. **Sentiment 情绪类（可选）**  
   - `news_sent_3d`: 基于新闻标题情绪的 3 日均值（AkShare 新闻 → bge 情绪打分）。

> **维度字段**：`symbol, date, industry_1, mktcap_log`（对数市值用于中性化）。

---

### C) 预处理流水线（Pipeline）
按 **交易日切片** 做横截面处理：
1. **对齐与补齐**：与交易日日历对齐；财务类特征日频化并 `ffill`，最多回填 65 个交易日；超限→`NaN`。
2. **去极值（Winsorize）**：`q1=1%/q99=99%`，或按 `Median±5*MAD`（鲁棒可选），横截面执行。
3. **中性化（Neutralize）**：对每个特征做 `industry_1` 与 `mktcap_log` 的 OLS 残差，得到 `*_neu`；
4. **标准化（Standardize）**：横截面 z-score，输出 `*_z`；
5. **滞后（Lag=1）**：写入前统一 `shift(1)`，绑定 `visibility_ts`；
6. **质检（QC）**：缺失率、偏度/峰度、横截面分布、IC 初筛，异常触发告警。

> 以上步骤的参数、窗口写入 `feature_meta`，支持回放与审计。

---

### D) 计算与调度（Compute & Schedule）
- **引擎**：Polars/Pandas（单机）→ Spark/Polars Lazy（后续扩展）。
- **并行**：按日期分片（DAG），或按行业桶并行；
- **调度**：Temporal/Prefect 任务流：`ingest → align → calc_raw → winsor → neutralize → zscore → lag → qc → publish`；
- **性能指标**：
  - 全量（5 年 × 5000 只 × 20 因子）≤ 30 分钟；
  - 增量（日更）≤ 3 分钟；
  - 峰值内存 < 8GB（分块处理）。

---

### E) 存储与模式（Storage & Schema）
- **列式主存**：`s3://naxs-features/<freq>/name=<factor>/date=YYYY-MM-DD/part-*.parquet`；
- **行式索引**：PostgreSQL（近期热数据与元数据）。

**`feature_meta`**  
```
(name, version, expr, window, neutralize_by, winsor(q1,q99|mad), scaler, lag, freq, created_at, created_by, doc)
```
**`feature_value`（PostgreSQL 热表）**  
```
(date, symbol, name, version, value, visibility_ts, quality_flags, industry_1, mktcap_log)
索引：(date, name, version), (symbol, date)
```

---

### F) API 合同（Contracts）
- `POST /factors/compute`
```json
{
  "universe": {"type": "index", "code": "000300.SH"},
  "features": [
    {"name": "pe_ttm", "version": "v1"},
    {"name": "mom_20", "version": "v1", "neutralize": true}
  ],
  "window": {"start": "2021-01-01", "end": "2025-09-17"},
  "freq": "D",
  "publish": true
}
```
**响应**（异步 Job）：`{ jobId }` → `GET /jobs/:id`：
```json
{
  "status": "succeeded",
  "rows": 1250000,
  "artifacts": [
    {"name":"pe_ttm","path":"s3://.../name=pe_ttm/date=..."}
  ],
  "trace_id": "tr_abc123"
}
```
- `GET /features/:name/meta`、`GET /features/:name/slice?date=2025-09-17`。
- **错误码**：`400` 参数非法、`409` 依赖缺失（如缺少 `eps_ttm`）、`422` 口径冲突、`503` 资源不足。

**事件（Kafka/NATS）**：
- `naxs.feature.built` `{name, version, rows, window, trace_id}`
- `naxs.feature.qc_failed` `{name, date, reason, samples, trace_id}`

---

### G) 质量评估（Validation & Reports）
- **横截面 IC/IR**：与次日收益 `ret_1d` 的 Spearman IC；
  - 报表：日度 IC、滚动均值（60D）、ICIR（年化），胜率（IC>0 的比例）。
- **分组测试（Q1..Q5）**：按特征分 5 组的多空收益；
- **稳定性**：缺失率、分布漂移（PSI）、行业/市值暴露；
- **阈值**（告警）
  - 缺失率 > 20%（非财务因子）  
  - |IC| < 0.01 且持续 60D  
  - PSI > 0.2（相较过去 1 年）

所有评估图表输出到 **报告中心**（可导出 PDF/CSV）。

---

### H) 自定义因子（Beta）
- **接口**：`POST /custom-factor`（异步 Job）
```json
{
  "name": "my_factor_v1",
  "dsl": {
    "expr": "zscore(winsorize(pct_change(adj_close, 20), 0.01, 0.99))",
    "neutralize": ["industry_1", "mktcap_log"],
    "lag": 1
  },
  "window": {"start": "2022-01-01", "end": "2025-09-17"}
}
```
- **安全沙箱**：
  - DSL 优先，若允许 Python 代码，执行于 **Firejail/容器** + 资源限额（CPU/内存/时限/IO）；
  - 禁止网络/文件系统写入；白名单库（numpy, polars, numba）。
- **审核**：需通过 QC 报告（IC、缺失率、分布）后才可 `publish=true` 对外可见。

---

### I) 知识图谱对接（选配）
- 节点：`Company, Industry, Indicator, Event, Factor`；
- 关系：`(Factor)-[MEASURES]->(Indicator)`、`(Company)-[BELONGS_TO]->(Industry)`；
- 用途：
  - 在研究页展示“因子→指标→公司”的可解释路径；
  - 结合 RAG 提供因子定义与口径说明的原文引用。

---

### J) 监控与告警（Monitoring）
- **指标**：构建时长、吞吐（rows/s）、内存峰值、QC 失败数、事件积压；
- **告警**：
  - Job 超时（> P95×2）；
  - QC 失败/因子失效；
  - S3/OSS 写入失败或延迟 > 60s。

---

### K) 与 AkShare 的字段映射（Example）
- 价量：`stock_zh_a_hist(symbol)` → `close, open, high, low, volume, adj_factor`；
- 财务：`stock_financial_abstract(symbol)` / `stock_financial_report_sina(symbol)` → `net_profit, equity, eps`；
- 交易日历：`tool_trade_date_hist_sina()`；
> 字段兼容层需维护 **Mapping 表** 与 **单元测试**，当上游字段变更时自动告警并回滚到缓存。

---

### L) 测试用例（QA）
- **单测**：
  - 公式正确性（封闭样本）、去极值/中性化/标准化的边界条件；
  - 交易日缺失、停牌、复权异常的稳健性；
- **集成**：
  - 以 50 支样本股 × 1 年窗口跑全链路，与“基准 CSV”逐字段对比；
- **回归**：
  - 任何 `feature_meta` 变更均触发重算/对比与差异报告。

---

### M) 验收标准（DoD）
- 元数据与算子流水线**可回放**（给定 `date`、`version` 可复现）；
- 因子质量报告（IC/IR/分组）可导出；
- API/事件稳定（错误率 < 0.5%）；
- 计算性能满足 SLA（全量 ≤ 30m、增量 ≤ 3m）。

**责任分工**：量化研究员（公式与 QC 指标）、数据工程师（流水线与存储）、后端开发（API/事件）、AI 工程师（解释链/RAG 对接）。

---

#### 3.2.3 模型与回测层（T3） 模型与回测层（T3）
**目标**：实现策略回测与绩效分析。  
**用户故事**：As a user, I want backtest results so that I can evaluate strategy performance.  
**开发任务**：
- T3-1 策略规则：Top-K 筛选。  
- T3-2 回测适配器：Qlib 集成、自研 fallback。  
- T3-3 指标计算：Return、Drawdown、Sharpe、WinRate。  
- T3-4 体检报告：自动生成优劣势文本。  
- T3-5 Beta：多策略对比、市况识别。  
**API**：POST /backtest/run。  
**责任**：量化研究员验证，后端实现。  
**时间**：Week 2 回测接口，Week 3 指标+报告。  

#### 3.2.4 Agent 层（T4）

**目标（Outcome）**：通过任务编排、多 Agent 协作与合规控制，实现“自然语言→意图→任务树→执行→结果”的闭环。确保输出可回放、可解释，并在 Demo 阶段完成研究/回测/报告的端到端链路。

**用户故事**：As a user, I want to ask questions in natural language and receive orchestrated, compliant research outputs.

---

### A) 架构组成
- **Planner**：LLM 模块，将自然语言解析为任务树；
- **Executor**：调用工具（数据、因子、回测、报告）；
- **Governor**：约束控制（置信度、合规检查、预算/时限）；
- **Trace Manager**：记录输入、参数、输出、来源，支持回放；
- **Reflection**：低置信度时自动重试/交叉验证。

---

### B) 工作流（Workflow）
1. 用户输入 Query（自然语言）；
2. Planner → 解析为任务链（e.g., fetch data → compute factors → run backtest → generate report）；
3. Executor → 依次调用微服务 API（/data, /factors, /backtest, /reports）；
4. Governor → 检查参数合规（时间范围、因子合法性）；低置信度 → 触发 Reflection；
5. Trace Manager → 记录所有调用（trace_id）；
6. 输出给前端（ResultCard/报告）。

---

### C) 技术选型
- **LLM**：开源模型（Qwen-14B/LLaMA2）或云端 API；
- **框架**：LangChain Agents + 自研 Tool Router；
- **调度**：Temporal/Prefect；
- **合规日志**：MongoDB/Elastic + OSS 归档。

---

### D) API 合同
- `POST /agent/orchestrate`
```json
{
  "query": "请帮我做一份关于000001的研究报告",
  "context": {"user_id": "u123", "role": "pro"}
}
```
**响应**：`{ jobId, status, trace_id }`

- `GET /agent/jobs/:id` → {status, steps, result, trace_id}

- **事件（Kafka）**：
  - `naxs.agent.tool_called` {tool, params, trace_id}
  - `naxs.agent.low_confidence` {reason, trace_id}
  - `naxs.agent.completed` {jobId, result_summary, trace_id}

---

### E) Trace & 回放
- **Trace 字段**：prompt, intent, tools, params, outputs, evidence, timestamps；
- **回放接口**：`GET /compliance/trace/:trace_id` → timeline；
- **水印**：报告/输出均带“非投资建议”与 trace_id。

---

### F) 验收标准（DoD）
- 自然语言输入可正确分解为任务树（准确率 ≥85%）；
- 每个执行均产生 trace，可回放；
- 低置信度任务触发 Reflection；
- Demo 支持：选股、回测、报告生成；
- 错误率 <1%。

**责任分工**：AI 工程师（Planner/Reflection）、后端开发（Executor/API）、合规专员（Governor/Trace）、产品经理（验收与用户体验）。

---

#### 3.2.5 应用层（T5） 应用层（T5）
**目标**：提供交互界面。  
**用户故事**：As a user, I want to interact with the system easily.  
**开发任务**：
- 前端：React + Tailwind，页面（研究、回测、报告中心）。  
- 后端：Express/Flask 路由，统一 API /api/v1/。  
**责任**：前端工程师开发 UI，后端提供接口。  
**时间**：Week 2 研究/回测页，Week 3 报告中心。  

#### 3.2.6 基础设施与安全（T6）
**目标**：可扩展与合规。  
**开发任务**：
- T6-1 API 网关：Kong，OAuth2/JWT。  
- T6-2 队列：Kafka/NATS。  
- T6-3 监控：Prometheus + Grafana。  
- T6-4 灾备：S3 备份，多 AZ 部署。  
- T6-5 合规：水印与免责声明模板。  
**责任**：DevOps 主导。  
**时间**：Week 1-2 基础设施 setup，持续优化。  

---

### 3.3 测试阶段（Week 4: Day 22-28）
**目标**：验证全链路。  
**活动**：
- 单元测试：覆盖率 >80%。
- 集成测试：Mock 端到端链路。  
- 系统测试：负载（Locust）、安全（OWASP）。  
- 用户测试：小范围 Beta 验证。  
**输出**：测试报告 + Bug 列表。

---

### 3.4 上线与维护阶段（Day 29-30+）
**目标**：部署 MVP Demo 并维护。  
**活动**：
- 部署：K8s to AWS，蓝绿发布。  
- 监控：警报配置。  
- 维护：Hotfix、版本迭代。  
**输出**：上线链接，运维手册。  

---

## 4. 长期规划与扩展
- **Beta 阶段**：增加自定义因子、仓位建议、实时信号订阅。
- **商业化阶段**：支持机构 API、团队协作、合规日志大规模应用。
- **未来扩展**：多市场支持（港股/美股）、多模态（语音/图像）、策略商城。


---

## 附录 A：3.1 数据层（深度规范・可落地）
**目标（Outcome）**：在 AkShare 为主的数据源前提下，构建稳定、可回放、可审计的数据采集与治理层，为上游因子/回测/Agent 提供口径一致、带 `visibility_ts` 的原始与衍生数据。

### A1. 数据域与表（Domains & Tables）
- **price_ohlcv（日频）**：`date, symbol, open, high, low, close, volume, amount, adj_factor, is_trading_day, visibility_ts, source, mapping_version`
- **fundamental（季度/年度，日频化）**：`period, symbol, revenue, net_profit, equity, eps_basic, roe, ... , visibility_ts, source, mapping_version`
- **security_master**：`symbol, name, list_date, delist_date, exchange, industry_1, is_st, visibility_ts, source`
- **trade_calendar**：`date, is_open`

### A2. 反前视与口径
- T+1 可见：`visibility_ts = trading_day 16:00:00 +08:00`；
- 复权口径：默认后复权，保留 `close_raw` 与 `adj_factor`；
- 停牌/特别处理：非交易日不写价量；ST/退市由 `security_master` 标记；
- 幸存者偏差：历史股票池按当期 `trade_calendar` 切片；
- 财务披露对齐：按披露日+1 个交易日滞后。

### A3. 采集与调度
- Python httpx/aiohttp + Celery/Temporal；
- 流程：`discover_symbols → fetch_prices → fetch_fundamentals → normalize → qc → upsert → partition → publish`；
- 幂等：自然键 + `ON CONFLICT DO UPDATE`；
- 限流与重试：并发≤4、`retry=3 backoff=2^n`；
- 缓存：Redis TTL=1d，冷备 CSV；
- 映射层：`akshare_mapping`（字段/单位/类型/示例/变更时间），变更触发事件与告警。

### A4. 质检（Great Expectations）
- 价量约束：`high>=max(open,close)`, `low<=min(open,close)`；
- 缺失率：日度 <1%（股票池横截面）；
- 分位/离群：体量分位在经验范围；
- 失败处理：写入 `data_anomalies` 与 `data_qc_failures`，发布 `naxs.data.qc_failed`。

### A5. 存储与分区
- 热数据：PostgreSQL（近 1 年），索引 `(date, symbol)`；
- 冷数据：OSS/S3 + Parquet（`s3://naxs-raw/domain=<price|fund|master>/date=YYYY-MM-DD/part-*.parquet`）。

### A6. API & 事件
- `GET /data/price?symbol=600519.SS&start=2022-01-01&end=2025-09-17`
- `GET /data/fundamental?symbol=600519.SS&period=ttm`
- `GET /meta/securities?date=2025-09-17`
- `GET /calendar/trading?start=...&end=...`
- 事件：`naxs.data.ingested`, `naxs.data.contract_violation`, `naxs.ingest.mapping_changed`
- 错误：`429` 限流，`424` 依赖失败，`503` 维护中。

### A7. 监控与 SLA
- 指标：抓取成功率、吞吐、字段变更次数、QC 失败数、API p95；
- SLA：历史全量 ≤90m；日增量 ≤10m；API p99 < 800ms（热数据）。

### A8. 测试与验收（DoD）
- 映射层单测覆盖≥80%；数据契约可回放；QC 报告可导出；事件与告警闭环。

---

## 附录 B：3.6 Agent 层（深度规范・可落地）
**目标（Outcome）**：将自然语言意图转为任务图（DAG），调用受控工具，产出可解释结果并记录可回放 Trace；支持阿里云上本地/云端开源 LLM（Qwen 系列）。

### B1. 架构分层
- Intent Parser → Planner → Governor → Executor → Explainer → Trace Logger。

### B2. Tool Registry
```json
{
  "name": "run_backtest",
  "version": "v1",
  "endpoint": "/backtest/run",
  "method": "POST",
  "schema": {"rules":{}, "universe":{}, "benchmark":{}},
  "timeout_ms": 900000,
  "rate_limit": {"rpm": 60},
  "retries": 2,
  "auth": "service"
}
```
- 分类：`data_query, factor_compute, screening, backtest, report_generate, compliance_trace`；
- 版本：后向兼容，新字段可选。

### B3. 任务图 Schema（Planner 输出）
```json
{
  "goal": "research_single_stock",
  "steps": [
    {"id":"s1","tool":"data_query","params":{"symbol":"600519.SS","window":"2y"}},
    {"id":"s2","tool":"factor_compute","depends_on":["s1"],"params":{"features":["pe_ttm","mom_20"]}},
    {"id":"s3","tool":"run_backtest","depends_on":["s2"],"params":{"rules":{}}},
    {"id":"s4","tool":"report_generate","depends_on":["s2","s3"],"params":{"type":"single_stock"}}
  ],
  "constraints": {"tn": true, "max_cost": 200, "timeout_ms": 120000}
}
```

### B4. Governor（守护）
- 白名单工具与参数校验；
- 口径校验：强制 `/contracts/data` 先行；
- 配额/限流：按用户与会话；
- 成本预算：超限降级窗口或停止；
- 低置信度：`confidence < 0.7` → 自动 `cross_check`（变更基准/窗口复算）。

### B5. 执行与状态机
- `created → planned → running → waiting → succeeded | failed | cancelled`；
- 幂等：`trace_id + step_id`，失败指数退避重试；
- 可中断/恢复：持久化任务图与中间结果。

### B6. 合规与 Trace
- Trace 字段：系统/用户提示、任务图、每步参数、输出摘要、数据来源与 `visibility_ts`、模型与因子版本、低置信度处理、最终答复；
- `GET /assistant/trace/:traceId` 回放；敏感信息脱敏。

### B7. API & 事件
- `POST /assistant/parse` → `{intent, tasks[], confidence}`
- `POST /assistant/execute` → `{jobId}`；`GET /assistant/jobs/:id` → `{status, result, trace_id}`
- `GET /assistant/tools` → 工具与 schema 列表
- 事件：`naxs.agent.tool_called`, `naxs.agent.low_confidence`

### B8. LLM（阿里云）
- 模型：Qwen2-7B/14B（CPU 可跑低并发）→ Qwen2-72B（GPU, PAI-DSW/vLLM）
- 推理：PAI-EAS 或自建 vLLM；函数调用输出 JSON 符合任务图 Schema；
- 缓存：Prompt 级 Redis Cache（Key=hash(prompt+context)）。

### B9. RAG/知识图谱
- 先 `semantic_search`（Top 5）→ 作为 Planner 上下文；
- 研究页展示 `(因子 → 指标 → 公司)` 的关系与来源链接。

### B10. 测试与验收
- 解析正确率 ≥ 95%（样例集）；
- 端到端成功率 ≥ 95%（单股研究、Top-K 回测、月报）；
- 非注册工具/越权 endpoint 请求被拒；
- 所有回答可在 Trace 中定位到证据与口径。


---

## 附录 C：工程工具链落地方案（Cursor + Traefik + 全栈 DevOps）
> 目标：给团队一套**可直接落地**的工程化方案，利用 Cursor 等 AI Coding 工具提效，并在本地/云端（阿里云）形成一致的开发-测试-部署流水线。

### C1. 仓库结构（Monorepo，pnpm + Python 多服务）
```
Naxs/
├─ apps/
│  ├─ web/                 # React + TS + Tailwind 前端
│  ├─ gateway/             # API 网关（Node/Express 或 Kong 配置）
│  └─ docs/                # 文档站（Docusaurus/Storybook）
├─ services/
│  ├─ data/                # 数据接入服务（Python FastAPI + AkShare）
│  ├─ features/            # 因子计算（Python FastAPI + Polars/Feast）
│  ├─ backtest/            # 回测服务（Python FastAPI + Qlib Adapter）
│  ├─ agent/               # Agent 编排（Python/Node + Temporal/Prefect）
│  └─ report/              # 报告生成（Python）
├─ packages/
│  ├─ ui/                  # 共享 UI 组件（React）
│  ├─ config/              # ESLint/Prettier/Tsconfig/Ruff 等共享配置
│  └─ sdk/                 # TypeScript/Python SDK（对 /api/v1 封装）
├─ infra/
│  ├─ devcontainer/        # VSCode/Cursor DevContainer
│  ├─ docker/              # 各服务 Dockerfile
│  ├─ compose/             # 本地 docker-compose 堆栈
│  ├─ traefik/             # 反向代理与本地路由
│  └─ k8s/                 # ACK 部署清单（Helm/Manifests）
├─ .cursorrules            # Cursor 约束与上下文规则
├─ .pre-commit-config.yaml # 统一代码质量钩子
├─ Makefile                # 一键任务（make up/test/lint）
├─ pnpm-workspace.yaml
└─ README.md
```

### C2. Cursor 配置（.cursorrules）
```yaml
# 目标：让 Cursor 生成的代码与本项目契约一致，避免“幻觉接口”。
project:
  name: Naxs
  principles:
    - 遵循 /api/v1 OpenAPI 契约，未知接口一律先 mock 再实现
    - TypeScript/Python 均启用严格模式（tsc --strict, ruff + mypy/pyright）
    - 所有计算遵循 T+1 可见性与 visibility_ts 口径
context:
  include:
    - packages/config/**/*.json
    - services/**/openapi.yaml
    - services/**/schemas/*.py
    - apps/web/src/**/*.ts*
rules:
  - 当用户请求“因子计算/回测”，先读取 services/features/backtest 的 OpenAPI
  - 返回的对象必须含 trace_id、version、visibility_ts（若适用）
  - 任何数据库写入必须幂等（ON CONFLICT DO UPDATE / upsert）
actions:
  scaffolds:
    - name: new-fastapi
      path: services/{{name}}
      template: fastapi
    - name: new-react-page
      path: apps/web/src/pages/{{name}}
      template: react-page
```

### C3. DevContainer（VSCode/Cursor 一致环境）
`infra/devcontainer/devcontainer.json`
```json
{
  "name": "naxs-dev",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": {"ghcr.io/devcontainers/features/node:1": {"version": "20"}},
  "postCreateCommand": "pnpm i --frozen-lockfile || npm i -g pnpm && pnpm i",
  "mounts": ["source=naxs_oss_cache,target=/workspace/.cache,type=volume"],
  "customizations": {"vscode": {"extensions": [
    "GitHub.copilot","ms-python.python","charliermarsh.ruff","ms-azuretools.vscode-docker"
  ]}}
}
```

### C4. 本地服务编排（docker-compose + Traefik）
`infra/compose/docker-compose.yml`
```yaml
version: "3.9"
services:
  traefik:
    image: traefik:v3.0
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
    ports: ["80:80", "8080:8080"]
    volumes: ["/var/run/docker.sock:/var/run/docker.sock:ro"]

  postgres:
    image: postgres:15
    environment: { POSTGRES_PASSWORD: naxs, POSTGRES_USER: naxs, POSTGRES_DB: naxs }
    ports: ["5432:5432"]

  redis:
    image: redis:7
    ports: ["6379:6379"]

  minio:
    image: minio/minio:latest
    environment: { MINIO_ROOT_USER: naxs, MINIO_ROOT_PASSWORD: naxspass }
    command: server /data --console-address ":9001"
    ports: ["9000:9000", "9001:9001"]

  data:
    build: ../../services/data
    labels:
      - "traefik.http.routers.data.rule=PathPrefix(`/api/v1/data`)"
      - "traefik.http.services.data.loadbalancer.server.port=8000"
    environment:
      - DATABASE_URL=postgresql://naxs:naxs@postgres:5432/naxs
      - REDIS_URL=redis://redis:6379/0

  features:
    build: ../../services/features
    labels:
      - "traefik.http.routers.features.rule=PathPrefix(`/api/v1/features`)"
      - "traefik.http.services.features.loadbalancer.server.port=8000"
    environment:
      - DATABASE_URL=postgresql://naxs:naxs@postgres:5432/naxs
      - OSS_ENDPOINT=http://minio:9000

  backtest:
    build: ../../services/backtest
    labels:
      - "traefik.http.routers.backtest.rule=PathPrefix(`/api/v1/backtest`)"
      - "traefik.http.services.backtest.loadbalancer.server.port=8000"

  agent:
    build: ../../services/agent
    labels:
      - "traefik.http.routers.agent.rule=PathPrefix(`/api/v1/assistant`)"
      - "traefik.http.services.agent.loadbalancer.server.port=8000"

  web:
    build: ../../apps/web
    labels:
      - "traefik.http.routers.web.rule=PathPrefix(`/`)"
      - "traefik.http.services.web.loadbalancer.server.port=3000"
```
> 说明：Traefik 负责把所有 `/api/v1/*` 路由到对应服务，前端域名直接落到 `web`。你可以通过 <http://localhost> 访问。

### C5. 质量门禁与 Git 流程
- **分支**：`main`（受保护）、`dev`、`feat/*`、`fix/*`
- **代码规范**：
  - TypeScript：ESLint（严格）+ Prettier + tsconfig `strict: true`
  - Python：ruff + black + mypy（或 pyright）
- **pre-commit**（`.pre-commit-config.yaml`）：
```yaml
repos:
- repo: https://github.com/charliermarsh/ruff-pre-commit
  rev: v0.6.8
  hooks: [{id: ruff, args: ["--fix"]}, {id: ruff-format}]
- repo: https://github.com/psf/black
  rev: 24.8.0
  hooks: [{id: black}]
- repo: https://github.com/pre-commit/mirrors-prettier
  rev: v4.0.0
  hooks: [{id: prettier}]
```

### C6. GitHub Actions（CI 示例）
`.github/workflows/ci.yml`
```yaml
name: ci
on: [push, pull_request]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: npm i -g pnpm && pnpm i
      - run: pnpm -C apps/web build && pnpm -C apps/web test
      - run: pip install -r services/data/requirements.txt
      - run: pytest -q
```

### C7. 前端快速脚手架（apps/web）
`apps/web/package.json`
```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "lint": "eslint .",
    "test": "vitest run",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18",
    "react-dom": "^18",
    "@tanstack/react-query": "^5",
    "axios": "^1",
    "zustand": "^4",
    "echarts": "^5"
  }
}
```

### C8. Python 服务通用模板（FastAPI）
`services/features/main.py`
```python
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Naxs Features")

class ComputeReq(BaseModel):
    universe: dict
    features: list
    window: dict
    freq: str = "D"
    publish: bool = True

@app.post("/api/v1/features/compute")
def compute(req: ComputeReq):
    # TODO: 拉取数据→计算→S3/PG 落库→返回 trace_id
    return {"status": "accepted", "jobId": "job_123", "trace_id": "tr_abc"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### C9. 一键命令（Makefile）
```makefile
up:        ## 启动本地栈（Traefik+PG+Redis+MinIO+服务+前端）
	docker compose -f infra/compose/docker-compose.yml up -d --build

down:      ## 停止并清理
	docker compose -f infra/compose/docker-compose.yml down -v

logs:      ## 查看所有服务日志
	docker compose -f infra/compose/docker-compose.yml logs -f --tail=200

test:      ## 运行前后端测试
	pnpm -C apps/web test && pytest -q
```

### C10. 迁移到阿里云（ACK）清单（概览）
- **镜像仓库**：阿里云 ACR；
- **命名空间**：`naxs-demo`；
- **Ingress**：Traefik/ALB Ingress Controller；
- **存储**：RDS Postgres、Redis 专有实例、OSS；
- **CI/CD**：GitHub Actions → ACR 推送 → ACK 滚动/蓝绿发布；
- **Secret/Config**：K8s Secret（数据库/Redis/OSS 凭证）、ConfigMap（服务配置）。

---

> 提示：如果你说的 “Trae” 指的是 **Traefik**，以上已完整覆盖；如果是其他工具（例如 Trace/Træ/Trunk），告诉我具体名称，我会追加对应的配置模板与工作流。



---

# 第四部分：功能闭环与耦合详解（可实施级）
> 目标：把每个模块拉到“能跑通”的粒度，明确上下游**强/弱耦合点**、接口契约、错误与补偿、SLA 与回放，确保端到端闭环可验收。

## 4.1 端到端闭环蓝图（E2E Loops）

### Loop A：单只股票研究闭环（Research → Report）
1) **前端研究页** 触发 → `GET /data/price`、`GET /data/fundamental`（Data 层）
2) **Features 服务** 计算快速横截面因子（缓存命中优先）→ `POST /features/compute?publish=false`
3) **Explainer** 生成因子贡献 & 口径说明 → 返回 JSON（含 `visibility_ts`、`trace_id`）
4) 用户点击 **一键报告** → `POST /reports/generate`（输入：数据片段 + 因子贡献 + 解释）
5) **Compliance** 写入 Trace → `GET /assistant/trace/:traceId` 可回放
6) 前端展示报告 + 下载（MD/PDF）

**耦合点**：
- 与 Data 的**语义耦合**：研究页所有指标必须携带 `visibility_ts`；
- 与 Features 的**性能耦合**：低延迟查询（p95 < 800ms）影响前端体验；
- 与 Compliance 的**审计耦合**：报告必须包含 `trace_id` 与免责声明。

---

### Loop B：选股 + 回测闭环（Screening → Backtest → Report）
1) 前端配置条件/权重 → `POST /screening/run`（Features/Screening）
2) 结果 Top-K → 用户点击“回测” → `POST /backtest/run`（Backtest）
3) 回测完成 → `naxs.backtest.completed` 事件 → 前端轮询 `GET /backtest/jobs/:id`
4) 体检报告（优势/风险/建议）→ `POST /reports/generate`
5) Trace 全链路记录（输入参数/指标/图表 URI）

**耦合点**：
- Screening 对 Features 的**数据口径耦合**：必须使用 `*_z` 或 `*_neu` 标准化后因子；
- Backtest 对 Screening 的**参数耦合**：Top-K/调仓频率/换手成本需一致；
- Report 对 Backtest 的**工件耦合**：图表/指标由回测服务产出（URI），报告只渲染。

---

### Loop C：因子构建闭环（Ingestion → Feature Build → QC → Publish）
1) Data 触发增量 → `naxs.data.ingested` 事件
2) Features 监听事件 → 计算原子因子 → `winsor → neutralize → zscore → lag`
3) QC（缺失率、IC 初筛）→ 通过则 `publish=true` 写入 OSS/PG
4) 发布 `naxs.feature.built` 事件 → Screening/Backtest 才可引用

**耦合点**：
- Data 与 Features 的**前视耦合**：`visibility_ts` 与 `lag=1` 必须检查；
- Features 与 Screening/Backtest 的**版本耦合**：引用时明确 `feature_meta.version`。

---

### Loop D：Agent 编排闭环（Parse → Plan → Execute → Explain → Trace）
1) 用户自然语言 → `POST /assistant/parse`（LLM 解析任务图）
2) Planner 产出 DAG → Governor 校验（白名单/口径/预算）
3) Executor 串并行调用 Tools（Data/Features/Backtest/Reports）
4) Explainer 组装输出（证据/口径/贡献）→ 前端渲染
5) 全链路 Trace 记录 → `GET /assistant/trace/:traceId` 回放

**耦合点**：
- 与 Tool Registry 的**接口耦合**：Schema 变化需要 Planner 兼容；
- 与 Compliance 的**追溯耦合**：每步参数/输出必须落 Trace；
- 与 LLM 的**成本/延迟耦合**：Governor 控制降级策略。

---

## 4.2 模块耦合矩阵（Coupling Matrix）
| 上游 | 下游 | 耦合类型 | 约束/契约 | 破坏性变更策略 |
|---|---|---|---|---|
| Data | Features | 数据口径/时间耦合 | `visibility_ts`, `lag=1`, 复权一致 | 版本字段 `mapping_version` + 回滚缓存 |
| Features | Screening | 统计口径耦合 | 仅可用 `*_z`/`*_neu`，窗口对齐 | `feature_meta.version` 必填；灰度发布 |
| Screening | Backtest | 参数/交易耦合 | Top-K、调仓频率、成本一致 | OpenAPI `x-compatible-with` 注记 |
| Backtest | Reports | 工件耦合 | 图表/指标由回测生成（URI） | 图表 Schema 加 `minor` 版本 |
| Planner | Tools | 接口耦合 | Tool Registry + JSON Schema | Planner 容错：忽略未知字段 |
| Any | Compliance | 审计耦合 | `trace_id`、`visibility_ts`、提示词/参数留痕 | Append-only，禁止覆盖 |

---

## 4.3 关键用例时序（文字版）
**用例 U1：Top-K → 回测 → 报告**
- 前端：`POST /screening/run` → `201 Accepted {trace_id}`
- Features：计算 → 返回 `list[{symbol, score, contrib[]}]`
- 前端：`POST /backtest/run`（携带 `trace_id` 贯穿）
- 回测：生成 `metrics/charts` → 事件 `naxs.backtest.completed`
- 前端：`POST /reports/generate`（附 `charts_uri` 和 `metrics`） → PDF URI
- Compliance：`trace` 完整保存（输入/中间/输出）

**用例 U2：自然语言研究 → 可解释报告**
- `POST /assistant/parse` → 任务图（Data→Features→Reports）
- Governor 校验口径（`/contracts/data`）→ Executor 执行
- Explainer 生成“因子-指标-公司”路径 + 来源链接
- Trace 可回放，报告带水印与免责声明

---

## 4.4 事务边界与幂等（Tx & Idempotency）
- **数据写入**：以自然键 upsert；失败重试不产生重复；
- **长任务**（回测/报告）：`jobId` 幂等键；状态机保证 `succeeded|failed|cancelled` 终态唯一；
- **事件**：使用 **exactly-once 语义** 近似（幂等消费 + 去重表 `event_outbox`）。

---

## 4.5 版本与兼容策略
- **数据映射**：`mapping_version` + 单测；发布前跑回归对比；
- **因子**：`feature_meta.version`；新版本灰度（并行保留旧版本 30 天）；
- **API**：`/api/v1` 固定，字段新增只加可选；破坏性变更需 `/v2` + Deprecation 公告。

---

## 4.6 失败恢复与补偿
- **数据采集**：失败 → 退避重试 → 冷备 CSV → 人工确认；
- **因子构建**：QC 失败 → 不 publish → 发告警与差异报告；
- **回测任务**：超时自动缩短窗口/降低分辨率重跑；
- **报告生成**：工件缺失则回退成纯文本摘要；
- **Agent**：低置信度触发 `cross_check`；未知工具调用直接拒绝并给出可选工具列表。

---

## 4.7 SLO / SLA 与告警
- **Data API** p95 < 800ms；**Features** 增量 ≤ 3m；**Backtest** 并发 200 任务成功率 ≥ 95%；
- **Agent** 端到端成功率 ≥ 95%；**Report** 30s 内完成率 ≥ 95%；
- 告警：超过阈值 5 分钟触发 Pager；连续 3 次失败自动升级严重级别。

---

## 4.8 契约校验与 CI Gate
- OpenAPI/JSONSchema 生成 **契约测试**（Dredd/Prism）；
- 因子/回测输出对比 **黄金样本**（Golden Master）；
- 事件契约：使用 AsyncAPI 文档校验 Producer/Consumer。

---

## 4.9 RACI（闭环责任）
| 闭环 | R（负责） | A（签核） | C（协作） | I（知会） |
|---|---|---|---|---|
| 单股研究 | 前端+Features | 产品 | Data/Compliance | 全员 |
| 选股回测 | Backtest | 技术负责人 | Features/前端 | 产品 |
| 因子构建 | 数据工程 | 量化负责人 | Backtest | 合规 |
| Agent 编排 | AI 工程 | 技术负责人 | Data/Features/Backtest | 合规 |
| 报告生成 | 报告服务 | 产品 | Backtest/Compliance | 前端 |

---

## 4.10 落地 Checklist（每个闭环上线前）
- [ ] 上游/下游接口在 **契约测试** 中通过（CI 必过）
- [ ] 事件已登记到 **Topic 白名单**（含 Schema）
- [ ] 性能压测报告（p95/p99）已归档
- [ ] Trace 可回放，报告含 `trace_id` 与水印
- [ ] 回滚方案（数据/服务）与演练记录

