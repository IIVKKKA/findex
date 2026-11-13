## 20 concurrent requests: `def` vs `async def` (1 worker)

| Route | Wall for 20 requests | Mean latency | /health p50 meanwhile | /health worst meanwhile |
|---|---|---|---|---|
| `/s_def` | 1061 ms | 871 ms | 71 ms | 514 ms |
| `/s_async` | 1141 ms | 695 ms | 7 ms | 686 ms |

## oha: 20 connections, 15 s, `GET /search?q=whale OR harpoon OR ship&k=10`

| Setup | Requests/s | p50, ms | p95, ms | p99, ms | 200 OK / total |
|---|---|---|---|---|---|
| 1 worker, `async def` | 19 | 1116 | 1734 | 2517 | 166/166 |
| 1 worker, `def` | 22 | 948 | 1646 | 1845 | 201/201 |
| 4 workers, `def` | 4 | 9273 | 9401 | 9401 | 20/20 |
