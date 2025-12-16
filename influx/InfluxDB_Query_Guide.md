# InfluxDB Query Guide for Factor Modeling

This guide covers querying factor data stored in InfluxDB using both the web interface and Python clients.

## Accessing InfluxDB

### Web Interface

- **URL**: http://localhost:8086
- **Username**: admin
- **Password**: factormodel2024
- **Organization**: qtab
- **Bucket**: jkp_factors

### Python Client Connection

```python
from influxdb_client_3 import InfluxDBClient3

# Connection for Docker environment
client = InfluxDBClient3(
    token="qtab_factor_token_2024_secure",
    host="http://localhost:8086",  # Use "influxdb:8086" inside containers
    database="jkp_factors",
    org="qtab"
)
```

## Data Schema

### Measurements

- **jkp_characteristics**: Main factor characteristics data
- **factor_returns**: Factor return series (if loaded)

### Tags (Indexed for Fast Queries)

- `gvkey`: Company identifier
- `permno`: CRSP permanent identifier
- `size_grp`: Size group classification
- `factor_model`: Data source/model identifier

### Fields (Numeric Values)

- `be_me`: Book-to-market ratio
- `ret_12_1`: 12-1 month momentum
- `gp_at`: Gross profits to assets
- `market_equity`: Market capitalization
- `beta_60m`: 60-month beta
- And 100+ other factor characteristics

## Query Examples

### Basic Data Retrieval

```sql
-- Get recent data for a specific company
SELECT be_me, ret_12_1, gp_at, market_equity
FROM jkp_characteristics
WHERE time >= '2024-01-01'
  AND gvkey = '001045'
ORDER BY time DESC

-- Get all data for specific characteristics
SELECT time, gvkey, be_me, ret_12_1
FROM jkp_characteristics
WHERE time >= '2023-01-01'
  AND be_me IS NOT NULL
  AND ret_12_1 IS NOT NULL
ORDER BY time DESC
```

### Cross-Sectional Analysis

```sql
-- Get latest cross-section of data
SELECT gvkey, be_me, ret_12_1, gp_at, market_equity
FROM jkp_characteristics
WHERE time = (
  SELECT MAX(time) FROM jkp_characteristics
)
AND be_me IS NOT NULL
ORDER BY market_equity DESC
LIMIT 500

-- Compare characteristics across size groups
SELECT size_grp,
       MEAN(be_me) as avg_book_to_market,
       MEAN(ret_12_1) as avg_momentum,
       COUNT(*) as count
FROM jkp_characteristics
WHERE time >= now() - 30d
GROUP BY size_grp
```

### Time Series Aggregations

```sql
-- Monthly aggregations by size group
SELECT MEAN(be_me) as avg_btm,
       STDDEV(ret_12_1) as vol_momentum,
       COUNT(*) as observations
FROM jkp_characteristics
WHERE time >= now() - 1y
GROUP BY time(1mo), size_grp
ORDER BY time DESC

-- Rolling correlations (using window functions)
SELECT time,
       CORR(be_me, ret_12_1) OVER (
         ORDER BY time
         ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
       ) as rolling_corr_12m
FROM jkp_characteristics
WHERE time >= '2023-01-01'
  AND be_me IS NOT NULL
  AND ret_12_1 IS NOT NULL
ORDER BY time
```

### Factor Portfolio Construction

```sql
-- Top and bottom deciles by book-to-market
WITH ranked_stocks AS (
  SELECT *,
         NTILE(10) OVER (
           PARTITION BY time
           ORDER BY be_me
         ) as btm_decile
  FROM jkp_characteristics
  WHERE time >= '2023-01-01'
    AND be_me IS NOT NULL
)
SELECT time, btm_decile,
       COUNT(*) as stock_count,
       MEAN(ret_12_1) as avg_momentum,
       MEAN(market_equity) as avg_market_cap
FROM ranked_stocks
WHERE btm_decile IN (1, 10)  -- Bottom and top deciles
GROUP BY time, btm_decile
ORDER BY time DESC, btm_decile
```

## Python Query Examples

### Using InfluxDB Client

```python
import pandas as pd
from influxdb_client_3 import InfluxDBClient3

# Initialize client (adjust host for your environment)
client = InfluxDBClient3(
    token="qtab_factor_token_2024_secure",
    host="http://localhost:8086",  # Use "influxdb:8086" inside Docker
    database="jkp_factors",
    org="qtab"
)

# Query recent data
query = """
SELECT time, gvkey, be_me, ret_12_1, gp_at
FROM jkp_characteristics
WHERE time >= now() - 90d
ORDER BY time DESC
"""

df = client.query(query)
print(f"Retrieved {len(df)} observations")
```

### Factor Analysis Queries

```python
# Get cross-sectional data for factor analysis
def get_cross_section(date='2024-01-31'):
    query = f"""
    SELECT gvkey, be_me, ret_12_1, gp_at, market_equity, ni_at
    FROM jkp_characteristics
    WHERE time = '{date}'
      AND be_me IS NOT NULL
      AND ret_12_1 IS NOT NULL
      AND gp_at IS NOT NULL
    ORDER BY market_equity DESC
    """
    return client.query(query)

# Get time series for specific companies
def get_company_time_series(gvkeys, start_date='2023-01-01'):
    gvkey_list = "', '".join(gvkeys)
    query = f"""
    SELECT time, gvkey, be_me, ret_12_1, gp_at, ret_exc
    FROM jkp_characteristics
    WHERE gvkey IN ('{gvkey_list}')
      AND time >= '{start_date}'
    ORDER BY gvkey, time
    """
    return client.query(query)

# Example usage
cross_section = get_cross_section('2024-01-31')
time_series = get_company_time_series(['001045', '001078', '001104'])
```

### Portfolio Analytics

```python
# Momentum portfolio analysis
def analyze_momentum_portfolios():
    query = """
    WITH momentum_deciles AS (
      SELECT *,
             NTILE(10) OVER (
               PARTITION BY time
               ORDER BY ret_12_1
             ) as mom_decile
      FROM jkp_characteristics
      WHERE time >= now() - 1y
        AND ret_12_1 IS NOT NULL
        AND ret_exc IS NOT NULL
    )
    SELECT time, mom_decile,
           COUNT(*) as stock_count,
           MEAN(ret_exc) as avg_return,
           STDDEV(ret_exc) as vol_return,
           MEAN(be_me) as avg_btm
    FROM momentum_deciles
    GROUP BY time, mom_decile
    ORDER BY time DESC, mom_decile
    """
    return client.query(query)

portfolio_stats = analyze_momentum_portfolios()
```

## Performance Optimization

### Indexing Strategy

- **Tags are automatically indexed**: Use for filtering (gvkey, size_grp, etc.)
- **Time is always indexed**: Time-based queries are very fast
- **Fields are not indexed**: Avoid complex WHERE clauses on field values

### Query Best Practices

```python
# ✅ GOOD: Filter by tags first, then fields
query = """
SELECT be_me, ret_12_1
FROM jkp_characteristics
WHERE gvkey = '001045'        -- Tag filter (fast)
  AND time >= '2023-01-01'    -- Time filter (fast)
  AND be_me > 0.5             -- Field filter (slower but necessary)
"""

# ❌ AVOID: Complex field calculations in WHERE
query = """
SELECT * FROM jkp_characteristics
WHERE be_me / market_equity > 0.001  -- Slow calculation
"""

# ✅ BETTER: Calculate after retrieval
query = """
SELECT be_me, market_equity
FROM jkp_characteristics
WHERE time >= '2023-01-01'
"""
# Then calculate be_me/market_equity in pandas
```

### Memory Management

```python
# For large queries, use chunking
def query_in_chunks(base_query, chunk_size_months=6):
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()

    results = []
    current_date = start_date

    while current_date < end_date:
        chunk_end = current_date + timedelta(days=chunk_size_months*30)

        chunk_query = f"""
        {base_query}
        AND time >= '{current_date.strftime('%Y-%m-%d')}'
        AND time < '{chunk_end.strftime('%Y-%m-%d')}'
        """

        chunk_data = client.query(chunk_query)
        results.append(chunk_data)
        current_date = chunk_end

    return pd.concat(results, ignore_index=True)
```

## Common Query Patterns

### Data Quality Checks

```sql
-- Check data availability by date
SELECT time, COUNT(*) as observation_count
FROM jkp_characteristics
GROUP BY time
ORDER BY time DESC
LIMIT 50

-- Check missing values
SELECT
  COUNT(*) as total_obs,
  COUNT(be_me) as be_me_count,
  COUNT(ret_12_1) as momentum_count,
  COUNT(gp_at) as profitability_count
FROM jkp_characteristics
WHERE time >= now() - 30d
```

### Factor Loadings

```sql
-- Estimate factor loadings using regression
WITH factor_data AS (
  SELECT time, gvkey,
         ret_exc,
         (ret_exc - AVG(ret_exc) OVER (PARTITION BY time)) as excess_ret,
         (be_me - AVG(be_me) OVER (PARTITION BY time)) as value_factor,
         (ret_12_1 - AVG(ret_12_1) OVER (PARTITION BY time)) as momentum_factor
  FROM jkp_characteristics
  WHERE time >= now() - 1y
    AND ret_exc IS NOT NULL
    AND be_me IS NOT NULL
    AND ret_12_1 IS NOT NULL
)
SELECT gvkey,
       REGR_SLOPE(excess_ret, value_factor) as value_loading,
       REGR_SLOPE(excess_ret, momentum_factor) as momentum_loading,
       REGR_R2(excess_ret, value_factor) as value_r2
FROM factor_data
GROUP BY gvkey
HAVING COUNT(*) >= 36  -- At least 3 years of data
```

## Troubleshooting

### Connection Issues

```python
# Test connection
try:
    result = client.query("SELECT COUNT(*) FROM jkp_characteristics LIMIT 1")
    print("✅ Connection successful")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### Query Debugging

```python
# Add query timing
import time

start_time = time.time()
result = client.query(query)
elapsed = time.time() - start_time

print(f"Query returned {len(result)} rows in {elapsed:.2f} seconds")
```

### Memory Issues

```python
# Check memory usage
import psutil

def check_memory():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}%")
    print(f"Available: {memory.available / 1024**3:.1f} GB")

check_memory()
```

This guide provides the foundation for querying factor data in InfluxDB. For advanced analytics, consider using the data in conjunction with pandas, numpy, and specialized financial libraries.
