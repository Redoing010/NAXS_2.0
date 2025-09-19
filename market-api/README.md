# NAXS Market Data API (Backend-first)

## 本地运行

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 3001 --reload
```

打开文档：`http://localhost:3001/docs`

## Docker 运行

```bash
docker build -t naxs-market-api .
docker run --rm -p 3001:3001 --env-file .env naxs-market-api
```

## 已实现接口

- `GET /api/spot` —— 全市场快照（分页+排序）
- `GET /api/minute` —— 单只分钟K（period: 1/5/15/30/60, adjust: qfq/hfq/None）
- `GET /api/bars` —— 直接读取 Parquet 仓（权威口径）
- `POST /api/admin/*` —— 触发拉取/DQ/Qlib 构建（需 `X-Admin-Token`）
- `GET /health`

## 示例调用

```bash
curl "http://localhost:3001/api/spot?sort_by=amount&descending=true&page=1&page_size=100"

curl "http://localhost:3001/api/minute?symbol=000001&period=1&adjust=qfq&limit=240"

curl "http://localhost:3001/api/bars?code=000831.SZ&start=2024-01-01&end=2024-12-31&freq=D"
```



