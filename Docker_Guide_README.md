# Docker Setup for QTAB Factor Modeling

This guide covers setting up the containerized environment for factor modeling
research using TimescaleDB/Postgres and Jupyter Lab.

> **Storage layer note.** This project used InfluxDB early on and has since moved
> to TimescaleDB. TimescaleDB is not a separate database — it is a PostgreSQL
> extension, so the `timescaledb` service is a normal Postgres 14 server on port
> 5432 that you query with normal SQL. Anything in the repo still referring to
> InfluxDB (`influx/`, `deprecated_docker/`) is obsolete.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose v2.0 or higher
- At least 4GB of available RAM
- WRDS account (for data access - each user must configure individually)

## Quick Start

### 1. Configure credentials

The stack will not start correctly without a `.env` file. It is gitignored, so a
fresh clone does not have one:

```bash
cp .env.template .env
# then edit .env and fill in WRDS_USERNAME / WRDS_PASSWORD
```

### 2. Start the Docker Environment

```bash
# Clone the repository and navigate to the project directory
cd factor-modeling

# Start all services (TimescaleDB + Jupyter Lab)
docker compose up -d

# View service status
docker compose ps

# View logs (useful for troubleshooting)
docker compose logs -f
```

### 3. Access Services

- **Jupyter Lab**: http://localhost:8888

  - Token: `qtab_jupyter_2024` (set by `JUPYTER_TOKEN` in `.env`)
  - Notebook directories are mounted under `/home/jovyan/work/`

- **TimescaleDB/Postgres**: `localhost:5432`
  - There is no web UI — connect with `psql` or any Postgres client
  - User / password / database: `PGUSER` / `PGPASSWORD` / `PGDATABASE` in `.env`
    (defaults `postgres` / `postgres` / `factor_db`)

```bash
# From the host
psql -h localhost -U postgres -d factor_db

# Or without a local psql install
docker exec -it factor_timescaledb psql -U postgres -d factor_db
```

### 4. Initial Setup

1. **Access Jupyter Lab** at http://localhost:8888
2. **Open the pipeline notebook** (`ryan/sample/run_pipeline.ipynb`)
3. **Configure your WRDS credentials** (see WRDS Configuration section below)
4. **Load and store factor data** by running the pipeline

The database starts **empty**. Table data lives in the `timescaledb_data` Docker
volume on the machine that created it — it is not in the repo and not in the
image. A new clone must either run the pipeline or restore a dump (see Data
Management).

## Service Configuration

### TimescaleDB Configuration

- **Image**: `timescale/timescaledb:latest-pg14` (PostgreSQL 14 + TimescaleDB extension)
- **Port**: 5432 (host) → 5432 (container)
- **Data persistence**: Docker volume `timescaledb_data`
- **Hostname from the jupyter container**: `timescaledb` (set in `docker-compose.yml`)
- **Hostname from the host**: `localhost`

### Jupyter Lab Configuration

- **Base Image**: `jupyter/scipy-notebook:python-3.11`
- **Port**: 8888 (host) → 8888 (container)
- **Working Directory**: `/home/jovyan/work`
- **Pre-installed packages**: wrds, psycopg2, SQLAlchemy, pandas, statsmodels, plotly, etc.

## WRDS Configuration

**Important**: Each user must configure their own WRDS credentials for security.

### Method 1: Environment Variables (Recommended)

```bash
# Copy the template and fill in your credentials (.env is gitignored)
cp .env.template .env

# Restart the container to pick up new environment variables
docker compose restart jupyter
```

### Method 2: Runtime Configuration

In your Jupyter notebook:

```python
import wrds
import getpass

# Configure WRDS connection
username = input("WRDS Username: ")
password = getpass.getpass("WRDS Password: ")

wrds_db = wrds.Connection(wrds_username=username, wrds_password=password)
```

## Directory Structure

```
factor-modeling/
├── docker-compose.yml          # Service orchestration
├── Dockerfile.jupyter          # Jupyter Lab container definition
├── .env.template               # Copy to .env and fill in
├── notebooks/                  # Shared notebooks (mounted in container)
├── ryan/                      # Ryan's notebooks (mounted in container)
├── alan/                      # Alan's notebooks (mounted in container)
├── shared-data/               # Shared data files (mounted in container)
├── src/                       # Source code for utilities
├── influx/                    # Obsolete — InfluxDB stack is deprecated
└── deprecated_docker/         # Obsolete — previous InfluxDB compose stack
```

## Common Commands

### Container Management

```bash
# Start services
docker compose up -d

# Stop services (will close local host ports)
docker compose down

# You can check to see if services are stopped by running the following in the terminal
# Check if ports are still in use
lsof -i :8888   # Jupyter
lsof -i :5432   # TimescaleDB/Postgres

#Stop specific services
docker compose stop jupyter

# Restart a specific service
docker compose restart jupyter

# View logs for a specific service
docker compose logs -f timescaledb

# Execute commands in running container
docker compose exec jupyter bash

# Dump timescaledb content in container onto local disk
docker exec factor_timescaledb pg_dump -U postgres -Fc factor_db > factor_db.dump

```

### Data Management

```bash
# Back up the database to a file on the host
docker exec factor_timescaledb pg_dump -U postgres -Fc factor_db > factor_db.dump

# Restore that dump into a running container
docker exec -i factor_timescaledb pg_restore -U postgres -d factor_db --clean --if-exists < factor_db.dump

# Clean up all data (WARNING: This deletes everything, including the DB volume!)
docker compose down -v
```

### Development

```bash
# Rebuild Jupyter container after package changes
docker compose build jupyter
docker compose up -d

# Install additional Python packages
docker compose exec jupyter pip install package_name
```

## Troubleshooting

### Common Issues

1. **Port already in use**

   ```bash
   # Check what's using the port
   lsof -i :5432  # for TimescaleDB/Postgres
   lsof -i :8888  # for Jupyter

   # Modify ports in docker-compose.yml if needed
   ```

   A local Postgres install on the host is the usual culprit for 5432.

2. **Container won't start**

   ```bash
   # Check container logs
   docker compose logs jupyter
   docker compose logs timescaledb

   # Restart Docker Desktop
   # Clear Docker cache: docker system prune
   ```

3. **WRDS connection issues**

   - Ensure your WRDS account is active
   - Check two-factor authentication setup
   - Verify credentials in container environment

4. **Database connection refused**

   ```bash
   # Wait for Postgres to finish initializing on first boot
   docker compose logs timescaledb

   # Check if services can communicate (the jupyter image has no `ping`)
   docker compose exec -T jupyter python -c \
     "import socket; socket.create_connection(('timescaledb', 5432), 5); print('reachable')"

   # Confirm the extension is loaded
   docker exec factor_timescaledb psql -U postgres -d factor_db \
     -c "SELECT extname, extversion FROM pg_extension;"
   ```

   Note the hostname differs by location: `timescaledb` from inside the jupyter
   container, `localhost` from the host. `.env` deliberately leaves `PGHOST`
   unset so each side resolves correctly.

### Performance Optimization

- **Memory**: Increase Docker Desktop memory allocation to 8GB+ for large datasets
- **CPU**: Enable multi-core processing in Docker Desktop preferences
- **Storage**: Use SSD storage for Docker volumes when possible

### Security Notes

- Change default passwords in production environments
- `docker-compose.yml` publishes 5432 on all interfaces with a default
  `postgres`/`postgres` login — anyone on your network can reach the database
- Use Docker secrets for sensitive credentials
- Never commit WRDS credentials to version control
- Credentials can also leak through notebook output — clear outputs before committing
- Consider using Docker BuildKit for improved build security

## Next Steps

1. Copy `.env.template` to `.env` and set your WRDS credentials
2. Explore the `ryan/sample/` pipeline (see `ryan/sample/README.md`)
3. Load your first dataset by running `ryan/sample/run_pipeline.ipynb`
4. Query the results with `psql` or `pandas.read_sql`

## Support

For technical issues:

1. Check the troubleshooting section above
2. Review container logs: `docker compose logs`
3. Consult the project documentation
4. Open an issue in the project repository
