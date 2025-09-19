# NAXS 2.0 - 新一代量化数据平台

## 项目概述

NAXS 2.0 是一个现代化的量化数据平台，提供完整的数据获取、存储、处理和分析解决方案。系统采用模块化架构，集成了先进的监控、错误处理和健康检查机制，确保高可用性和数据一致性。

## 核心特性

### 🚀 数据处理
- **多源数据集成**: 支持AkShare等多种数据源
- **高效存储**: 基于Parquet格式的列式存储
- **实时更新**: 支持增量数据更新和实时数据流
- **数据质量**: 内置数据质量检查和清洗机制

### 📊 量化集成
- **Qlib兼容**: 无缝集成Qlib量化框架
- **统一接口**: 标准化的数据访问接口
- **回测支持**: 完整的历史数据回测能力
- **因子工程**: 内置常用技术指标和因子

### 🔧 系统监控
- **实时监控**: CPU、内存、磁盘等系统资源监控
- **错误追踪**: 完整的错误记录和分析
- **健康检查**: 多层次的系统健康状态检查
- **告警机制**: 智能告警和通知系统

### 🛡️ 可靠性保障
- **重试机制**: 智能重试和退避策略
- **熔断保护**: 防止级联故障的熔断器
- **优雅降级**: 服务异常时的降级处理
- **数据一致性**: 确保数据完整性和一致性

## 系统架构

```
NAXS 2.0
├── 数据层 (modules/data/)
│   ├── 数据源接口 (akshare_source.py)
│   ├── 存储引擎 (parquet_store.py)
│   ├── 交易日历 (calendar.py)
│   ├── 数据质量 (dq.py)
│   └── Qlib集成 (qlib_writer.py)
├── API层 (market-api/)
│   ├── 数据接口 (bars.py)
│   ├── 管理接口 (admin.py)
│   └── 健康检查 (health.py)
├── 监控层 (modules/monitoring/)
│   ├── 系统监控 (system_monitor.py)
│   ├── 错误处理 (error_handler.py)
│   └── 健康检查 (health_checks.py)
├── 操作脚本 (ops/)
│   ├── 数据拉取 (pull_prices.py)
│   ├── 质量检查 (dq_report.py)
│   └── Qlib构建 (build_qlib_bundle.py)
└── 配置管理 (configs/)
    ├── 数据配置 (data.yaml)
    └── 日志配置 (logging.yaml)
```

## 快速开始

### 环境要求

- Python 3.8+
- 8GB+ RAM
- 50GB+ 磁盘空间

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/Redoing010/NAXS_2.0.git
cd NAXS_2.0

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动系统

```bash
# 方式1: 使用系统启动脚本（推荐）
python start_system.py

# 方式2: 手动启动API服务
cd market-api
uvicorn app.main:app --host 0.0.0.0 --port 3001 --reload
```

### 验证安装

```bash
# 运行端到端测试
python test_e2e.py

# 检查系统健康状态
curl http://localhost:3001/health/detailed
```

## 使用指南

### 数据拉取

```bash
# 拉取指定股票的历史数据
python ops/pull_prices.py --codes 000001.SZ,600519.SH --start 2024-01-01 --end 2024-12-31

# 拉取所有A股数据（增量更新）
python ops/pull_prices.py --all --incremental
```

### 数据质量检查

```bash
# 生成数据质量报告
python ops/dq_report.py --codes 000001.SZ --format html --output reports/dq_report.html

# 批量质量检查
python ops/dq_report.py --sample 100 --format json
```

### Qlib数据包构建

```bash
# 构建Qlib数据包
python ops/build_qlib_bundle.py --codes 000001.SZ,600519.SH --start 2024-01-01

# 构建全市场数据包
python ops/build_qlib_bundle.py --all --validate
```

### API使用示例

```python
import requests

# 获取股票数据
response = requests.get('http://localhost:3001/api/bars', params={
    'code': '000001.SZ',
    'start': '2024-01-01',
    'end': '2024-12-31',
    'freq': 'D'
})
data = response.json()

# 检查系统健康状态
health = requests.get('http://localhost:3001/health/system').json()
print(f"系统状态: {health['overall_status']}")
```

## 监控和运维

### 系统监控

系统提供多层次的监控能力：

1. **实时指标监控**
   - CPU使用率
   - 内存使用情况
   - 磁盘空间
   - 网络流量

2. **应用层监控**
   - API响应时间
   - 错误率统计
   - 数据处理性能
   - 任务执行状态

3. **业务监控**
   - 数据新鲜度
   - 数据质量指标
   - 交易日历更新
   - Qlib集成状态

### 健康检查端点

```bash
# 基础健康检查
curl http://localhost:3001/health

# 详细健康检查
curl http://localhost:3001/health/detailed

# 系统指标
curl http://localhost:3001/health/metrics

# 错误统计
curl http://localhost:3001/health/errors
```

### 错误处理

系统内置了完善的错误处理机制：

- **重试机制**: 自动重试失败的操作
- **熔断保护**: 防止级联故障
- **优雅降级**: 部分功能异常时的降级处理
- **错误追踪**: 完整的错误日志和上下文

## 配置说明

### 数据配置 (configs/data.yaml)

```yaml
data_sources:
  akshare:
    enabled: true
    timeout: 30
    retry_count: 3

storage:
  parquet:
    root_path: "data/parquet"
    compression: "snappy"
    
qlib:
  data_path: "data/qlib"
  calendar_path: "data/qlib/calendars"
```

### 日志配置 (configs/logging.yaml)

```yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
handlers:
  console:
    class: logging.StreamHandler
    formatter: default
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/naxs.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
```

## 开发指南

### 代码结构

- `modules/data/`: 数据处理核心模块
- `modules/monitoring/`: 监控和错误处理模块
- `market-api/`: FastAPI应用
- `ops/`: 运维脚本
- `configs/`: 配置文件
- `docs/`: 技术文档

### 开发规范

1. **代码风格**: 遵循PEP 8规范
2. **类型注解**: 使用Python类型提示
3. **文档字符串**: 完整的函数和类文档
4. **错误处理**: 使用统一的错误处理机制
5. **日志记录**: 合理的日志级别和信息

### 测试

```bash
# 运行单元测试
pytest tests/

# 运行端到端测试
python test_e2e.py

# 代码覆盖率
pytest --cov=modules tests/
```

## 部署指南

### Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 3001

CMD ["python", "start_system.py"]
```

### 生产环境配置

1. **资源配置**
   - CPU: 4核心以上
   - 内存: 16GB以上
   - 磁盘: SSD，100GB以上

2. **网络配置**
   - 确保外网访问（数据源）
   - 配置防火墙规则
   - 负载均衡（如需要）

3. **监控配置**
   - 配置告警通知
   - 设置监控阈值
   - 日志收集和分析

## 故障排除

### 常见问题

1. **数据拉取失败**
   - 检查网络连接
   - 验证AkShare版本
   - 查看错误日志

2. **API响应慢**
   - 检查系统资源
   - 优化查询参数
   - 检查数据索引

3. **监控异常**
   - 检查依赖包安装
   - 验证权限配置
   - 查看系统日志

### 日志位置

- 系统日志: `logs/system.log`
- API日志: `logs/api.log`
- 错误日志: `logs/error.log`
- 监控日志: `logs/monitoring.log`

## 性能优化

### 数据处理优化

1. **并行处理**: 使用多进程处理大批量数据
2. **内存管理**: 分批处理避免内存溢出
3. **缓存策略**: 合理使用缓存减少重复计算
4. **索引优化**: 为常用查询建立索引

### API性能优化

1. **异步处理**: 使用异步IO提高并发
2. **连接池**: 复用数据库连接
3. **响应压缩**: 启用gzip压缩
4. **缓存策略**: 缓存热点数据

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交代码变更
4. 编写测试用例
5. 提交Pull Request

## 许可证

MIT License

## 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [Issues]
- 技术讨论: [Discussions]

---

**注意**: 本项目仅供学习和研究使用，投资有风险，请谨慎决策