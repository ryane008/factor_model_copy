# InfluxDB Enterprise Home Setup Guide

This guide provides step-by-step instructions for setting up InfluxDB Enterprise Home for the factor modeling project.

## Option 1: InfluxDB Enterprise Home Installation (macOS)

### 1. Download InfluxDB Enterprise Home

1.  curl -O https://www.influxdata.com/d/install_influxdb3.sh && sh install_influxdb3.sh enterprise
2. simple download
3. quickstart
4. Make sure to select Home free option!

### 2. Create license and token for enterprise home
1. provide email and verify to create free license
2. source ~/.zshrc (reload shell)
3. influxdb3 create token --admin
4. in env


### 2. Installation

- Extract the downloaded file
- Run the installer:
  ```bash
  sudo installer -pkg influxdb-enterprise-home-*.pkg -target /
  ```

### 3. Start InfluxDB

Choose one of the following methods:

**Method 1: Using systemctl**

```bash
sudo systemctl start influxdb
```

**Method 2: Using launchctl**

```bash
sudo launchctl load /Library/LaunchDaemons/com.influxdata.influxdb.plist
```

### 4. Configuration

- **Config file location**: `/usr/local/etc/influxdb/influxdb.conf`
- **Default ports**:
  - HTTP: 8086
  - RPC: 8088
- **Default data directory**: `/usr/local/var/lib/influxdb/`

### 5. Create Database

Choose one of the following methods:

**Method 1: Web Interface**

- Access: http://localhost:8086
- Create database through the UI

**Method 2: CLI**

```bash
influx -execute "CREATE DATABASE jkp_factors"
```

### 6. Python Integration

Install the required Python package:

```bash
pip install influxdb3-python
```

## Option 2: Docker Setup (Recommended for Development)

### 1. Pull InfluxDB Image

```bash
docker pull influxdb:latest
```

### 2. Run Container

```bash
docker run -d -p 8086:8086 -p 8088:8088 \
  -v influxdb-storage:/var/lib/influxdb \
  -e INFLUXDB_DB=jkp_factors \
  -e INFLUXDB_ADMIN_USER=admin \
  -e INFLUXDB_ADMIN_PASSWORD=password \
  influxdb:latest
```

### 3. Access

- Web Interface: http://localhost:8086
- Username: admin
- Password: password

## Project-Specific Configuration

For this factor modeling project, we're using a custom setup with:

- **Host**: `localhost:8181`
- **Database**: `jkp_factors`
- **Token**: Generated via API (see notebook for token management)

## Verification

To verify your installation is working:

1. **Check if service is running**:

   ```bash
   # For native installation
   sudo systemctl status influxdb

   # For Docker
   docker ps | grep influxdb
   ```

2. **Test connection**:

   ```bash
   curl -i http://localhost:8086/ping
   ```

3. **List databases**:
   ```bash
   influx -execute "SHOW DATABASES"
   ```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Make sure ports 8086 and 8088 are not in use by other services
2. **Permission issues**: Run installation commands with sudo
3. **Service not starting**: Check logs in `/var/log/influxdb/`

### Useful Commands

```bash
# Stop InfluxDB
sudo systemctl stop influxdb

# Restart InfluxDB
sudo systemctl restart influxdb

# Check InfluxDB logs
tail -f /var/log/influxdb/influxd.log

# Uninstall InfluxDB (if needed)
sudo systemctl stop influxdb
sudo rm -rf /usr/local/bin/influx*
sudo rm -rf /usr/local/var/lib/influxdb
```

## Security Notes

- Change default passwords in production
- Configure SSL/TLS for production environments
- Restrict network access to InfluxDB ports
- Regularly backup your databases

## Performance Optimization

For large datasets like the JKP factor data:

1. **Increase batch size** in write operations
2. **Use appropriate retention policies**
3. **Configure memory and disk settings** in influxdb.conf
4. **Use tags efficiently** for better query performance

## Additional Resources

- [InfluxDB Documentation](https://docs.influxdata.com/)
- [InfluxDB Python Client](https://github.com/influxdata/influxdb-client-python)
- [Time Series Best Practices](https://docs.influxdata.com/influxdb/v2.0/write-data/best-practices/)
