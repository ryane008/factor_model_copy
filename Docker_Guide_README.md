# Docker Setup for QTAB Factor Modeling

This guide covers setting up the containerized environment for factor modeling research using InfluxDB and Jupyter Lab.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose v2.0 or higher
- At least 4GB of available RAM
- WRDS account (for data access - each user must configure individually)

## Quick Start

### 1. Start the Docker Environment

```bash
# Clone the repository and navigate to the project directory
cd factor-modeling

# Start all services (InfluxDB + Jupyter Lab)
docker-compose up -d

# View service status
docker-compose ps

# View logs (useful for troubleshooting)
docker-compose logs -f
```

### 2. Access Services

- **Jupyter Lab**: http://localhost:8888

  - Token: `qtab_jupyter_2024`
  - All notebooks are mounted in `/work/` directory

- **InfluxDB Web UI**: http://localhost:8086
  - Username: `admin`
  - Password: `factormodel2024`
  - Organization: `qtab`
  - Bucket: `jkp_factors`

### 3. Initial Setup

1. **Access Jupyter Lab** at http://localhost:8888
2. **Open the Docker data loader notebook** (`docker_data_access.ipynb`)
3. **Configure your WRDS credentials** (see WRDS Configuration section below)
4. **Load and store factor data** using the provided data loader

## Service Configuration

### InfluxDB Configuration

- **Version**: InfluxDB 2.7
- **Port**: 8086 (host) → 8086 (container)
- **Data persistence**: Docker volume `influxdb_data`
- **Admin Token**: `qtab_factor_token_2024_secure`

### Jupyter Lab Configuration

- **Base Image**: `jupyter/scipy-notebook:latest`
- **Port**: 8888 (host) → 8888 (container)
- **Working Directory**: `/home/jovyan/work`
- **Pre-installed packages**: wrds, influxdb-client, statsmodels, plotly, etc.

## WRDS Configuration

**Important**: Each user must configure their own WRDS credentials for security.

### Method 1: Environment Variables (Recommended)

```bash
# Create a .env file in the project root (this file should NOT be committed)
echo "WRDS_USERNAME=your_username" > .env
echo "WRDS_PASSWORD=your_password" >> .env

# Restart the container to pick up new environment variables
docker-compose restart jupyter
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
├── notebooks/                  # Shared notebooks (mounted in container)
├── ryan/                      # Ryan's notebooks (mounted in container)
├── alan/                      # Alan's notebooks (mounted in container)
├── shared-data/               # Shared data files (mounted in container)
└── src/                       # Source code for utilities
```

## Common Commands

### Container Management

```bash
# Start services
docker-compose up -d

# Stop services (will close local host ports)
docker-compose down

# You can check to see if services are stopped by running the following in the terminal
# Check if ports are still in use
lsof -i :[jupyter port]
lsof -i :[influxdb port]

#Stop specific services
docker-compose stop jupyter

# Restart a specific service
docker-compose restart jupyter

# View logs for a specific service
docker-compose logs -f influxdb

# Execute commands in running container
docker-compose exec jupyter bash

```

### Data Management

```bash
# Backup InfluxDB data
docker-compose exec influxdb influx backup /tmp/backup
docker cp factor_influxdb:/tmp/backup ./influxdb_backup

# Clean up all data (WARNING: This deletes everything!)
docker-compose down -v
```

### Development

```bash
# Rebuild Jupyter container after package changes
docker-compose build jupyter
docker-compose up -d

# Install additional Python packages
docker-compose exec jupyter pip install package_name
```

## Troubleshooting

### Common Issues

1. **Port already in use**

   ```bash
   # Check what's using the port
   lsof -i :8086  # for InfluxDB
   lsof -i :8888  # for Jupyter

   # Modify ports in docker-compose.yml if needed
   ```

2. **Container won't start**

   ```bash
   # Check container logs
   docker-compose logs jupyter
   docker-compose logs influxdb

   # Restart Docker Desktop
   # Clear Docker cache: docker system prune
   ```

3. **WRDS connection issues**

   - Ensure your WRDS account is active
   - Check two-factor authentication setup
   - Verify credentials in container environment

4. **InfluxDB connection refused**

   ```bash
   # Wait for InfluxDB to fully initialize (can take 30-60 seconds)
   docker-compose logs influxdb

   # Check if services can communicate
   docker-compose exec jupyter ping influxdb
   ```

### Performance Optimization

- **Memory**: Increase Docker Desktop memory allocation to 8GB+ for large datasets
- **CPU**: Enable multi-core processing in Docker Desktop preferences
- **Storage**: Use SSD storage for Docker volumes when possible

### Security Notes

- Change default passwords in production environments
- Use Docker secrets for sensitive credentials
- Never commit WRDS credentials to version control
- Consider using Docker BuildKit for improved build security

## Next Steps

1. Read the [InfluxDB Query Guide](InfluxDB_Query_Guide.md)
2. Explore the `docker_data_access.ipynb` notebook
3. Set up your WRDS credentials
4. Load your first dataset using the Docker-compatible data loader

## Support

For technical issues:

1. Check the troubleshooting section above
2. Review container logs: `docker-compose logs`
3. Consult the project documentation
4. Open an issue in the project repository
