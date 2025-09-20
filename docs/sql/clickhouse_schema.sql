-- NAXS 2.0 canonical ClickHouse schema
-- Generated from docs/schema.md

CREATE TABLE IF NOT EXISTS security_master
(
    symbol String COMMENT '股票代码，格式XXXXXX.XX',
    name String COMMENT '股票名称',
    exchange LowCardinality(String) COMMENT '交易所，SH/SZ',
    list_date Date COMMENT '上市日期',
    delist_date Nullable(Date) COMMENT '退市日期',
    industry_1 Nullable(String) COMMENT '一级行业',
    industry_2 Nullable(String) COMMENT '二级行业',
    is_st UInt8 DEFAULT 0 COMMENT '是否ST股票，0或1',
    market_cap Float64 COMMENT '总市值（元）',
    float_market_cap Float64 COMMENT '流通市值（元）',
    visibility_ts DateTime('UTC') COMMENT '数据可见时间',
    source String COMMENT '数据源'
)
ENGINE = MergeTree
PARTITION BY exchange
ORDER BY symbol
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS price_ohlcv_daily
(
    symbol String COMMENT '股票代码，格式XXXXXX.XX',
    exchange LowCardinality(String) DEFAULT splitByChar('.', symbol)[2] COMMENT '交易所，自动从代码解析',
    date Date COMMENT '交易日期',
    open Float64 COMMENT '开盘价',
    high Float64 COMMENT '最高价',
    low Float64 COMMENT '最低价',
    close Float64 COMMENT '收盘价',
    volume UInt64 COMMENT '成交量（股）',
    amount Float64 COMMENT '成交额（元）',
    adj_factor Float64 DEFAULT 1.0 COMMENT '复权因子',
    pct_chg Float64 COMMENT '涨跌幅（%）',
    change Float64 COMMENT '涨跌额（元）',
    turnover Float64 COMMENT '换手率（%）',
    amplitude Float64 COMMENT '振幅（%）',
    is_trading_day UInt8 DEFAULT 1 COMMENT '是否交易日，0或1',
    visibility_ts DateTime('UTC') COMMENT '数据可见时间',
    source String COMMENT '数据源',
    mapping_version String COMMENT '字段映射版本',
    CONSTRAINT c_price_daily_high_ge_open_close CHECK high >= greatest(open, close),
    CONSTRAINT c_price_daily_low_le_open_close CHECK low <= least(open, close),
    CONSTRAINT c_price_daily_amount_non_negative CHECK amount >= 0
)
ENGINE = MergeTree
PARTITION BY (exchange, toYYYYMM(date))
ORDER BY (symbol, date)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS price_ohlcv_minute
(
    symbol String COMMENT '股票代码，格式XXXXXX.XX',
    exchange LowCardinality(String) DEFAULT splitByChar('.', symbol)[2] COMMENT '交易所，自动从代码解析',
    datetime DateTime('UTC') COMMENT '时间戳',
    period UInt16 COMMENT '周期（分钟）',
    open Float64 COMMENT '开盘价',
    high Float64 COMMENT '最高价',
    low Float64 COMMENT '最低价',
    close Float64 COMMENT '收盘价',
    volume UInt64 COMMENT '成交量（股）',
    amount Float64 COMMENT '成交额（元）',
    visibility_ts DateTime('UTC') COMMENT '数据可见时间',
    source String COMMENT '数据源',
    CONSTRAINT c_price_minute_high_ge_open_close CHECK high >= greatest(open, close),
    CONSTRAINT c_price_minute_low_le_open_close CHECK low <= least(open, close),
    CONSTRAINT c_price_minute_valid_period CHECK period IN (1, 5, 15, 30, 60)
)
ENGINE = MergeTree
PARTITION BY (exchange, period, toYYYYMMDD(datetime))
ORDER BY (symbol, datetime)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS fundamental_quarterly
(
    symbol String COMMENT '股票代码，格式XXXXXX.XX',
    period String COMMENT '报告期，格式YYYY-QX',
    period_year UInt16 DEFAULT toUInt16(substring(period, 1, 4)) COMMENT '报告年份',
    report_date Date COMMENT '报告发布日期',
    revenue Float64 COMMENT '营业收入（万元）',
    net_profit Float64 COMMENT '净利润（万元）',
    total_assets Float64 COMMENT '总资产（万元）',
    net_assets Float64 COMMENT '净资产（万元）',
    eps_basic Float64 COMMENT '基本每股收益（元）',
    eps_diluted Float64 COMMENT '稀释每股收益（元）',
    roe Float64 COMMENT '净资产收益率（%）',
    roa Float64 COMMENT '总资产收益率（%）',
    debt_ratio Float64 COMMENT '资产负债率（%）',
    visibility_ts DateTime('UTC') COMMENT '数据可见时间',
    source String COMMENT '数据源'
)
ENGINE = MergeTree
PARTITION BY period_year
ORDER BY (symbol, period)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS trading_calendar
(
    date Date COMMENT '日期',
    is_open UInt8 COMMENT '是否开市，0或1',
    exchange LowCardinality(String) DEFAULT 'ALL' COMMENT '交易所代码，默认ALL',
    holiday_name Nullable(String) COMMENT '节假日名称',
    created_at DateTime('UTC') COMMENT '记录创建时间',
    source String COMMENT '数据源'
)
ENGINE = MergeTree
ORDER BY date
SETTINGS index_granularity = 8192;
