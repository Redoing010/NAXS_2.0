# NAXS 2.0 Runbook

## ClickHouse Schema 部署

1. 阅读 [`docs/schema.md`](docs/schema.md) 了解字段定义与质量校验规则。
2. 使用 [`docs/sql/clickhouse_schema.sql`](docs/sql/clickhouse_schema.sql) 中的DDL在目标ClickHouse集群创建或更新表结构。
   - 命令示例（默认通过本地 `clickhouse-client` 执行）：

```bash
clickhouse-client --multiline --multiquery < docs/sql/clickhouse_schema.sql
```

> 以上命令假设已在执行环境中安装并配置好 `clickhouse-client`。如需连接远程实例，请按需追加 `--host`、`--secure`、`--user`、`--password` 等参数。

3. 完成建表后，可通过以下检查确认表存在：

```sql
SHOW TABLES FROM <database> LIKE 'price_ohlcv_%';
```

如需调整字段或约束，请同时更新 `docs/schema.md` 与 `docs/sql/clickhouse_schema.sql`，确保文档和实际DDL保持一致。
