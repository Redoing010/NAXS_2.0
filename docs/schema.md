# NAXS数据模式规范

本文档定义了NAXS系统中所有数据的标准模式和字段规范，确保数据的一致性和可追溯性。

## 1. 数据标准化原则

### 1.1 命名规范
- **字段名称**：使用小写字母和下划线，如 `open_price`、`trading_date`
- **股票代码**：统一格式为 `XXXXXX.XX`，如 `000001.SZ`、`600519.SH`
- **日期格式**：ISO 8601格式 `YYYY-MM-DD`
- **时间戳**：UTC时区，ISO 8601格式 `YYYY-MM-DDTHH:MM:SS.sssZ`

### 1.2 数据类型规范
- **价格数据**：float64，精度到分（0.01）
- **成交量**：int64，单位为股
- **成交额**：float64，单位为元
- **日期**：date类型或datetime64[ns]
- **股票代码**：string类型

### 1.3 时区处理
- **存储时区**：所有时间数据以UTC时区存储
- **输入时区**：默认为Asia/Shanghai，自动转换为UTC
- **输出时区**：API返回UTC时间，前端根据需要转换本地时区

## 2. 核心数据模式

### 2.1 股票基础信息 (Security Master)

```yaml
table: security_master
partition: exchange
index: symbol

fields:
  symbol:           # 股票代码
    type: string
    format: "XXXXXX.XX"
    example: "000001.SZ"
    required: true
    
  name:             # 股票名称
    type: string
    max_length: 50
    example: "平安银行"
    required: true
    
  exchange:         # 交易所
    type: string
    enum: ["SH", "SZ"]
    example: "SZ"
    required: true
    
  list_date:        # 上市日期
    type: date
    format: "YYYY-MM-DD"
    example: "1991-04-03"
    required: true
    
  delist_date:      # 退市日期
    type: date
    format: "YYYY-MM-DD"
    nullable: true
    
  industry_1:       # 一级行业
    type: string
    max_length: 50
    example: "银行"
    
  industry_2:       # 二级行业
    type: string
    max_length: 50
    example: "股份制银行"
    
  is_st:           # 是否ST股票
    type: boolean
    default: false
    
  market_cap:      # 总市值（元）
    type: float64
    min_value: 0
    
  float_market_cap: # 流通市值（元）
    type: float64
    min_value: 0
    
  visibility_ts:    # 数据可见时间
    type: datetime
    timezone: "UTC"
    required: true
    
  source:          # 数据源
    type: string
    example: "akshare"
    required: true
```

### 2.2 日频价量数据 (Daily OHLCV)

```yaml
table: price_ohlcv_daily
partition: [exchange, date]
index: [symbol, date]

fields:
  symbol:           # 股票代码
    type: string
    format: "XXXXXX.XX"
    required: true
    
  date:             # 交易日期
    type: date
    format: "YYYY-MM-DD"
    required: true
    
  open:             # 开盘价
    type: float64
    precision: 2
    min_value: 0
    required: true
    
  high:             # 最高价
    type: float64
    precision: 2
    min_value: 0
    required: true
    
  low:              # 最低价
    type: float64
    precision: 2
    min_value: 0
    required: true
    
  close:            # 收盘价
    type: float64
    precision: 2
    min_value: 0
    required: true
    
  volume:           # 成交量（股）
    type: int64
    min_value: 0
    required: true
    
  amount:           # 成交额（元）
    type: float64
    precision: 2
    min_value: 0
    
  adj_factor:       # 复权因子
    type: float64
    precision: 6
    default: 1.0
    
  pct_chg:          # 涨跌幅（%）
    type: float64
    precision: 4
    
  change:           # 涨跌额（元）
    type: float64
    precision: 2
    
  turnover:         # 换手率（%）
    type: float64
    precision: 4
    min_value: 0
    
  amplitude:        # 振幅（%）
    type: float64
    precision: 4
    min_value: 0
    
  is_trading_day:   # 是否交易日
    type: boolean
    default: true
    
  visibility_ts:    # 数据可见时间（T+1）
    type: datetime
    timezone: "UTC"
    required: true
    
  source:           # 数据源
    type: string
    example: "akshare"
    required: true
    
  mapping_version:  # 字段映射版本
    type: string
    example: "v1.0"
    required: true

# 数据约束
constraints:
  - high >= max(open, close)  # 最高价约束
  - low <= min(open, close)   # 最低价约束
  - volume >= 0               # 成交量非负
  - amount >= 0               # 成交额非负
```

### 2.3 分钟频价量数据 (Minute OHLCV)

```yaml
table: price_ohlcv_minute
partition: [exchange, date, period]
index: [symbol, datetime]

fields:
  symbol:           # 股票代码
    type: string
    format: "XXXXXX.XX"
    required: true
    
  datetime:         # 时间戳
    type: datetime
    timezone: "UTC"
    required: true
    
  period:           # 周期（分钟）
    type: int
    enum: [1, 5, 15, 30, 60]
    required: true
    
  open:             # 开盘价
    type: float64
    precision: 2
    min_value: 0
    required: true
    
  high:             # 最高价
    type: float64
    precision: 2
    min_value: 0
    required: true
    
  low:              # 最低价
    type: float64
    precision: 2
    min_value: 0
    required: true
    
  close:            # 收盘价
    type: float64
    precision: 2
    min_value: 0
    required: true
    
  volume:           # 成交量（股）
    type: int64
    min_value: 0
    required: true
    
  amount:           # 成交额（元）
    type: float64
    precision: 2
    min_value: 0
    
  visibility_ts:    # 数据可见时间
    type: datetime
    timezone: "UTC"
    required: true
    
  source:           # 数据源
    type: string
    required: true
```

### 2.4 财务数据 (Fundamental Data)

```yaml
table: fundamental_quarterly
partition: [period_year]
index: [symbol, period]

fields:
  symbol:           # 股票代码
    type: string
    format: "XXXXXX.XX"
    required: true
    
  period:           # 报告期
    type: string
    format: "YYYY-QX"
    example: "2024-Q1"
    required: true
    
  report_date:      # 报告发布日期
    type: date
    format: "YYYY-MM-DD"
    required: true
    
  revenue:          # 营业收入（万元）
    type: float64
    precision: 2
    
  net_profit:       # 净利润（万元）
    type: float64
    precision: 2
    
  total_assets:     # 总资产（万元）
    type: float64
    precision: 2
    min_value: 0
    
  net_assets:       # 净资产（万元）
    type: float64
    precision: 2
    
  eps_basic:        # 基本每股收益（元）
    type: float64
    precision: 4
    
  eps_diluted:      # 稀释每股收益（元）
    type: float64
    precision: 4
    
  roe:              # 净资产收益率（%）
    type: float64
    precision: 4
    
  roa:              # 总资产收益率（%）
    type: float64
    precision: 4
    
  debt_ratio:       # 资产负债率（%）
    type: float64
    precision: 4
    min_value: 0
    max_value: 100
    
  visibility_ts:    # 数据可见时间
    type: datetime
    timezone: "UTC"
    required: true
    
  source:           # 数据源
    type: string
    required: true
```

### 2.5 交易日历 (Trading Calendar)

```yaml
table: trading_calendar
index: date

fields:
  date:             # 日期
    type: date
    format: "YYYY-MM-DD"
    required: true
    
  is_open:          # 是否开市
    type: boolean
    required: true
    
  exchange:         # 交易所
    type: string
    enum: ["SH", "SZ", "ALL"]
    default: "ALL"
    
  holiday_name:     # 节假日名称
    type: string
    nullable: true
    
  created_at:       # 创建时间
    type: datetime
    timezone: "UTC"
    required: true
    
  source:           # 数据源
    type: string
    required: true
```

## 3. 数据质量规范

### 3.1 数据完整性检查

```yaml
completeness_rules:
  missing_data:
    threshold: 0.05  # 最大缺失率5%
    severity: "warning"
    
  required_fields:
    - symbol
    - date/datetime
    - open
    - high
    - low
    - close
    - volume
    
  date_continuity:
    max_gap_days: 5  # 最大连续缺失5个交易日
    severity: "warning"
```

### 3.2 数据一致性检查

```yaml
consistency_rules:
  ohlc_logic:
    - high >= max(open, close)
    - low <= min(open, close)
    - all_prices > 0
    
  volume_logic:
    - volume >= 0
    - amount >= 0
    
  date_logic:
    - date <= current_date
    - no_future_dates
    - trading_days_only
```

### 3.3 数据准确性检查

```yaml
accuracy_rules:
  outlier_detection:
    price_change:
      method: "zscore"
      threshold: 3.0
      
    volume_spike:
      method: "iqr"
      multiplier: 5.0
      
  cross_validation:
    multiple_sources: true
    tolerance: 0.01  # 1%容差
```

## 4. API响应格式

### 4.1 标准响应格式

```json
{
  "code": "000001.SZ",
  "freq": "D",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "total_records": 245,
  "rows": [
    {
      "datetime": "2024-01-02T00:00:00Z",
      "open": 10.50,
      "high": 10.80,
      "low": 10.45,
      "close": 10.75,
      "volume": 12345678,
      "amount": 132456789.50
    }
  ],
  "metadata": {
    "source": "akshare",
    "last_updated": "2024-01-03T02:00:00Z",
    "data_quality": "good"
  }
}
```

### 4.2 错误响应格式

```json
{
  "error": {
    "code": "DATA_NOT_FOUND",
    "message": "No data found for 000001.SZ in range 2024-01-01 to 2024-12-31",
    "details": {
      "symbol": "000001.SZ",
      "start_date": "2024-01-01",
      "end_date": "2024-12-31",
      "available_range": {
        "start": "2020-01-01",
        "end": "2023-12-31"
      }
    },
    "timestamp": "2024-01-03T10:30:00Z"
  }
}
```

## 5. 存储格式规范

### 5.1 Parquet文件结构

```
data/parquet/
├── D/                    # 日频数据
│   ├── SH/              # 上交所
│   │   ├── 600000.parquet
│   │   └── 600519.parquet
│   └── SZ/              # 深交所
│       ├── 000001.parquet
│       └── 000002.parquet
├── 1m/                  # 1分钟数据
│   ├── SH/
│   │   └── 2024-01-02/
│   │       ├── 600000.parquet
│   │       └── 600519.parquet
│   └── SZ/
│       └── 2024-01-02/
│           ├── 000001.parquet
│           └── 000002.parquet
└── metadata/            # 元数据
    ├── symbols.parquet
    ├── calendar.parquet
    └── data_quality.parquet
```

### 5.2 Qlib格式规范

```
data/qlib_cn_daily/
├── calendars/
│   └── day.txt          # 交易日历
├── instruments/
│   └── all.txt          # 股票列表
├── features/
│   ├── sh600000/
│   │   ├── open.bin
│   │   ├── high.bin
│   │   ├── low.bin
│   │   ├── close.bin
│   │   ├── volume.bin
│   │   └── date.bin
│   └── sz000001/
│       ├── open.bin
│       └── ...
└── metadata.json        # 元数据
```

## 6. 版本控制

### 6.1 模式版本
- **当前版本**：v1.0
- **向后兼容**：支持v1.x所有版本
- **升级策略**：渐进式升级，保持API兼容性

### 6.2 变更日志

```yaml
v1.0 (2024-01-01):
  - 初始版本
  - 定义核心数据模式
  - 支持日频和分钟频数据
  
v1.1 (计划中):
  - 添加期权数据支持
  - 增强财务数据字段
  - 优化存储格式
```

## 7. 数据治理

### 7.1 数据血缘
- **源头追踪**：每条数据记录source字段
- **处理链路**：记录数据处理的每个步骤
- **影响分析**：变更影响范围评估

### 7.2 数据安全
- **访问控制**：基于角色的数据访问权限
- **审计日志**：记录所有数据访问和修改
- **数据脱敏**：敏感数据自动脱敏处理

### 7.3 合规要求
- **数据保留**：按监管要求保留历史数据
- **隐私保护**：遵循数据保护法规
- **免责声明**：所有输出数据包含免责声明