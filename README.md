# QTAB Factor Modeling Project

## Overview

This project provides a comprehensive framework for factor modeling research using the Jensen, Kelly, and Pedersen (JKP) global factor database. The system includes Docker containerization for easy setup, InfluxDB for time-series data storage, and Jupyter notebooks for analysis.

## � Environment Setup (.env Configuration)

### Prerequisites

- WRDS account with access to the JKP Global Factor database
- Docker Desktop installed and running

### Step 1: Create Your Environment File

```bash
# Copy the template file
cp .env.template .env

# Edit with your credentials (use any text editor)
nano .env
# OR
code .env
```

### Step 2: Configure Your WRDS Credentials

Edit the `.env` file and replace the placeholder values. The full annotated
version lives in `.env.template`; the values that matter are:

```bash
# REQUIRED — your WRDS credentials
WRDS_USERNAME=your_actual_wrds_username
WRDS_PASSWORD=your_actual_wrds_password

# TimescaleDB / Postgres — these both seed the database container on first boot
# and are what the notebooks connect with, so the two sides stay in sync
PGUSER=postgres
PGPASSWORD=postgres
PGDATABASE=factor_db
PGPORT=5432

# Jupyter
JUPYTER_TOKEN=qtab_jupyter_2024
JUPYTER_PORT=8888
```

The storage layer is **TimescaleDB/Postgres**. The InfluxDB variables that used
to live here are obsolete — that stack is in `deprecated_docker/`.

#### Why there is no `PGHOST`

The database hostname differs depending on where the code runs, and it resolves
with no branching and no entry in `.env`:

- **Inside the jupyter container**, `docker-compose.yml` sets
  `PGHOST=timescaledb` under `environment:`, which takes precedence over
  `env_file:`.
- **Outside the container** (local venv, host Jupyter, a plain script), `PGHOST`
  is unset and the code defaults to `localhost`, where compose publishes 5432.

So `os.getenv("PGHOST", "localhost")` covers every case on every OS. Setting
`PGHOST` in `.env` would work today only because compose happens to override it
— leave it out. Set it explicitly only if your database is somewhere other than
this compose stack.

This replaces the old `is_docker_environment()` helper, which sniffed
`JUPYTER_SERVER_ROOT`. A *local* Jupyter also sets that variable, so running a
notebook on the host under Jupyter would resolve `PGHOST` to `timescaledb` and
fail to connect.

### Step 3: Security Verification

✅ **Verify the .env file is ignored by git:**

```bash
git status          # .env must NOT appear in untracked files
```

⚠️ **CRITICAL SECURITY REMINDERS:**

- Never commit `.env` files to version control
- Each team member creates their own `.env` file
- **Never print a credential to a notebook cell.** `.gitignore` protects `.env`,
  but it cannot protect notebook *output* — a debug `print` of
  `WRDS_PASSWORD` in `ryan/charactersitics_loader.ipynb` was committed in
  `2d4b502`, which is how the current WRDS password leaked. Clear notebook
  outputs before committing, and rotate any credential that has ever been
  printed.

### Step 4: Test Your Configuration

```bash
# Start the Docker environment
docker compose up -d

# Check that the container can read the environment
docker compose exec jupyter env | grep WRDS_USERNAME
docker compose exec jupyter env | grep PGHOST     # should print timescaledb

# Confirm the database is reachable from the container
docker compose exec jupyter python -c "
import os, sqlalchemy as sa
e = sa.create_engine(f\"postgresql+psycopg2://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}\"
                     f\"@{os.getenv('PGHOST','localhost')}:{os.getenv('PGPORT','5432')}/{os.environ['PGDATABASE']}\")
print(e.connect().execute(sa.text('SELECT version()')).scalar())
"
```

## �🚀 Quick Start (Updated for .env)

### Prerequisites

- Docker Desktop installed and running
- 8GB+ RAM recommended
- For data access: WRDS account (each user configures individually)

### Complete Setup Process

```bash
# 1. Clone the repository
git clone <repository-url>
cd factor-modeling

# 2. Set up your credentials
cp .env.template .env
# Edit .env with your WRDS username and password

# 3. Start the Docker environment
docker-compose up -d

# 4. Wait for services to initialize (60 seconds)
sleep 60

# 5. Check service status
docker-compose ps
```

### Alternative: Using the Setup Script

```bash
# Automated setup with guided credential configuration
./start_docker.sh
```

### Verify Your Setup

1. **InfluxDB UI**: http://localhost:8086

   - Username: `admin`
   - Password: `factormodel2024`
   - Organization: `qtab`
   - Bucket: `jkp_factors`

2. **Jupyter Lab**: http://localhost:8888

   - Token: `qtab_jupyter_2024`
   - Your existing notebooks will automatically use .env credentials

3. **Test WRDS Connection**: Open `ryan/data_loading copy.ipynb` and run the first few cells

### Updating Credentials

```bash
# Edit your .env file
nano .env

# Restart containers to pick up changes
docker-compose restart

# Verify new credentials are loaded
docker-compose exec jupyter env | grep WRDS_USERNAME
```

### Access Points

- **Jupyter Lab**: http://localhost:8888 (token: `qtab_jupyter_2024`)
- **InfluxDB Web UI**: http://localhost:8086 (admin/factormodel2024)

## 📊 Data Pipeline Architecture

### 1. Data Sources

- **Primary**: WRDS JKP Global Factor Database (`contrib.global_factor`)
- **Alternative**: Shared CSV files in Docker environment
- **Frequency**: Monthly end-of-month data
- **Coverage**: Global equities with 200+ factor characteristics

### 2. Processing Layer

- **Docker Containers**: Isolated, reproducible environment
- **Python Libraries**: pandas, numpy, statsmodels, influxdb-client
- **Data Validation**: Automated checks for missing values, outliers
- **Factor Model Mapping**: Pre-configured academic factor definitions

### 3. Storage Layer

- **InfluxDB**: Optimized time-series database
- **Schema**: Measurements (jkp_characteristics), Tags (gvkey, permno), Fields (factor values)
- **Indexing**: Automatic indexing on time and tag columns
- **Retention**: Configurable data retention policies

### 4. Analysis Layer

- **Jupyter Notebooks**: Interactive analysis environment
- **Pre-built Models**: Fama-French, Quality, Momentum, Custom factors
- **Visualization**: Plotly, Matplotlib integration
- **Export**: Results to CSV, images, reports

## 🏗️ Directory Structure

```
factor-modeling/
├── 🐳 Docker Configuration
│   ├── docker-compose.yml          # Service orchestration
│   ├── Dockerfile.jupyter          # Jupyter Lab container
│   └── start_docker.sh            # Quick start script
│
├── 📓 Notebooks
│   ├── docker_data_access.ipynb    # Docker-compatible data loader
│   ├── factor_model.ipynb          # Factor model estimation
│   ├── characteristics.ipynb       # Cross-sectional analysis
│   └── data_loading.ipynb          # WRDS data extraction
│
├── 🔧 Source Code
│   ├── docker_data_loader.py       # Docker-compatible utilities
│   └── [other utilities]
│
├── 📁 Data Directories
│   ├── shared-data/                # Shared CSV files (Docker mount)
│   ├── ryan/                       # Ryan's notebooks
│   │   └── sample/                 # ⭐ Barra prototype on a 220-name sample
│   └── alan/                       # Alan's notebooks
│
└── 📚 Documentation
    ├── .env.template               # Annotated environment config
    ├── Docker_Setup_README.md      # Detailed Docker setup
    ├── InfluxDB_Query_Guide.md     # Obsolete — InfluxDB stack is deprecated
    └── JKP_Characteristics_README.md # Factor definitions
```

## ⭐ Barra risk model prototype — `ryan/sample/`

The active line of work. Builds the inputs for a USE4-style risk model on a
stratified sample of **20 securities per GICS sector × 11 sectors = 220**, drawn
from the securities alive at 2005-12-31 and followed forward.

```bash
docker compose up -d
# → http://localhost:8888 → ryan/sample/run_pipeline.ipynb, run top to bottom

python ryan/sample/test_logic.py     # 40 offline checks, no WRDS or database needed
```

Writes three tables, all prefixed `ryan_sample_` so they never collide with the
full-universe tables: `ryan_sample_universe` (the roster and the seed that
produced it), `ryan_sample_panel` (monthly characteristics keyed on `(id, eom)`),
and `ryan_sample_exposures` (standardized style factors and industry dummies).

**Why a sample:** the factor covariance matrix `F` needs `T >> K` monthly
cross-sections, not large `N`. The full-universe pull returned three months with
returns in only one of them, making `F` rank-1. A small universe with long
history is the correct trade.

See [`ryan/sample/README.md`](ryan/sample/README.md) for the design, the defects
it fixes in the full-universe pipeline, and the limitations it deliberately
accepts.

## 🎯 Factor Models Included

### Fama-French 5-Factor Model

```python
'fama_french_5': {
    'value': ['be_me', 'ni_me', 'ocf_me'],        # Book-to-market, earnings, cash flow
    'size': ['market_equity'],                     # Market capitalization
    'momentum': ['ret_12_1'],                      # 12-1 month momentum
    'profitability': ['gp_at', 'ni_at'],          # Gross profits, ROA
    'investment': ['at_gr1', 'oaccruals_at'],     # Asset growth, accruals
    'market': ['ret_exc', 'beta_60m']             # Market exposure
}
```

### Quality-Comprehensive Model

```python
'quality_comprehensive': {
    'profitability': ['gp_at', 'ni_at', 'ni_be', 'ope_be', 'ocf_at'],
    'quality_composite': ['f_score', 'qmj', 'qmj_prof', 'qmj_safety'],
    'earnings_quality': ['ni_ar1', 'oaccruals_at'],
    'growth_quality': ['sale_gr1', 'at_gr1', 'qmj_growth'],
    'financial_strength': ['at_be', 'cash_at']
}
```

### Other Available Models

- **Basic 4-Factor**: Returns, size, value, momentum
- **Trading Frictions**: Size, volatility, liquidity, beta
- **Academic Research**: Comprehensive 8-factor model
- **Custom Models**: Define your own factor combinations

## 🔐 Data Access Configuration

### Secure WRDS Integration with .env

Your data loader now automatically detects and uses WRDS credentials from your `.env` file:

```python
# Your existing notebook code now works seamlessly!
# No need to manually input credentials every time

# 1. WRDS connection is automatic
wrds_db = get_wrds_connection()  # Uses .env credentials

# 2. InfluxDB connection is Docker-aware
client = InfluxDBClient3(...)    # Uses Docker container network

# 3. Data loading works as before
data_loader = JKPDataLoader(wrds_db)
usa_data = data_loader.load_characteristics_data(
    factor_model='fama_french_5',
    months_back=12,
    countries=['USA']
)
```

### Environment Detection

One env lookup, no heuristics:

- **🐳 Inside the container**: `docker-compose.yml` sets `PGHOST=timescaledb`
- **💻 On the host**: `PGHOST` is unset, so the code defaults to `localhost:5432`
- **🔐 WRDS credentials**: read from `.env`; missing values raise with
  instructions rather than prompting

There is deliberately **no interactive credential fallback**. A prompt makes a
notebook non-reproducible and encourages pasting secrets into cells, which is
how the current WRDS password ended up in git history.

### Example .env Configuration

```bash
# Required
WRDS_USERNAME=your_username
WRDS_PASSWORD=your_password

# Database — seeds the container and is what notebooks connect with
PGUSER=postgres
PGPASSWORD=postgres
PGDATABASE=factor_db
PGPORT=5432
# PGHOST intentionally omitted — see "Why there is no PGHOST" above
```

## 📈 Usage Examples

### Loading Factor Data

```python
# Docker environment
data_loader, influxdb = setup_docker_environment()
characteristics = data_loader.load_characteristics("characteristics.csv")
factor_data = data_loader.filter_factor_model(characteristics, "fama_french_5")

# Store in InfluxDB
influxdb.write_characteristics(factor_data)
```

### Querying Stored Data

```python
# Get recent cross-section
query = """
SELECT gvkey, be_me, ret_12_1, gp_at, market_equity
FROM jkp_characteristics
WHERE time >= now() - 30d
ORDER BY market_equity DESC
LIMIT 500
"""
cross_section = influxdb.query(query)
```

### Factor Portfolio Analysis

```python
# Momentum decile portfolios
momentum_query = """
WITH momentum_deciles AS (
  SELECT *, NTILE(10) OVER (PARTITION BY time ORDER BY ret_12_1) as mom_decile
  FROM jkp_characteristics
  WHERE time >= now() - 1y AND ret_12_1 IS NOT NULL
)
SELECT time, mom_decile, COUNT(*) as stocks, MEAN(ret_exc) as avg_return
FROM momentum_deciles
GROUP BY time, mom_decile
ORDER BY time DESC, mom_decile
"""
portfolios = influxdb.query(momentum_query)
```

## 🛠️ Development Workflow

### 1. Environment Setup

```bash
# Start development environment
./start_docker.sh

# Or manually
docker-compose up -d
docker-compose logs -f
```

### 2. Data Preparation

```bash
# Place data files in shared directory
cp your_data.csv shared-data/

# Verify data accessibility
docker-compose exec jupyter ls -la /home/jovyan/work/shared-data/
```

### 3. Analysis Development

```bash
# Access Jupyter Lab
open http://localhost:8888

# Create new notebook or use existing templates
# - docker_data_access.ipynb for data loading
# - factor_model.ipynb for factor analysis
```

### 4. Production Deployment

```bash
# Export results
docker-compose exec jupyter jupyter nbconvert --to html your_analysis.ipynb

# Backup data
docker-compose exec influxdb influx backup /tmp/backup
```

## 🔧 Troubleshooting

### Common Issues

**Containers Won't Start**

```bash
# Check Docker status
docker info

# View logs
docker-compose logs

# Clean and restart
docker-compose down -v
docker-compose up -d
```

**Data Loading Errors**

```bash
# Verify file location
docker-compose exec jupyter ls -la /home/jovyan/work/shared-data/

# Check file format
docker-compose exec jupyter head /home/jovyan/work/shared-data/your_file.csv
```

**InfluxDB Connection Issues**

```bash
# Test connectivity
docker-compose exec jupyter ping influxdb

# Check InfluxDB logs
docker-compose logs influxdb

# Verify service status
curl http://localhost:8086/health
```

### Performance Optimization

- **Memory**: Increase Docker Desktop memory to 8GB+
- **CPU**: Enable multi-core processing in Docker preferences
- **Storage**: Use SSD for Docker volumes
- **Batch Size**: Adjust InfluxDB write batch sizes for large datasets

## 📚 Documentation

- **[Docker Setup Guide](Docker_Setup_README.md)**: Complete Docker configuration and troubleshooting
- **[InfluxDB Query Guide](InfluxDB_Query_Guide.md)**: Query patterns, performance tips, examples
- **[JKP Characteristics Guide](JKP_Characteristics_README.md)**: Factor definitions and academic references

## 🤝 Contributing

### Adding New Factor Models

1. Update `FACTOR_MODELS` dictionary in `docker_data_loader.py`
2. Validate characteristics exist in your dataset
3. Test with sample data
4. Document factor definitions and academic references

### Custom Analysis Templates

1. Create notebook in appropriate user directory
2. Use standardized data loading patterns
3. Include visualization and export functionality
4. Add documentation and usage examples

## 📞 Support

### Getting Help

1. **Documentation**: Check guides in this repository
2. **Logs**: Review container logs with `docker-compose logs`
3. **Issues**: Open GitHub issue with error details and logs
4. **Community**: Join discussions in project channels

### Reporting Issues

Include the following information:

- Operating system and Docker version
- Complete error messages and logs
- Steps to reproduce the issue
- Expected vs. actual behavior

---

**🎉 Happy Factor Modeling!** This framework provides a solid foundation for academic and professional factor research. Start with the basic examples and gradually explore more advanced features as your research develops.
