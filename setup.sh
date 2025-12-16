#!/bin/bash

echo "🚀 Starting Factor Modeling InfluxDB Environment"
echo "================================================"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop first."
    exit 1
fi

echo "✅ Docker found"

# Create directories
mkdir -p notebooks shared-data

# Stop any existing containers
docker-compose down 2>/dev/null || true

# Start InfluxDB and Jupyter
echo "🔄 Starting InfluxDB and Jupyter..."
docker-compose up -d

echo "⏳ Waiting for InfluxDB to start (30 seconds)..."
sleep 30

# Check services
echo "🔍 Checking services..."
if curl -f http://localhost:8086/ping > /dev/null 2>&1; then
    echo "✅ InfluxDB is running"
else
    echo "⚠️  InfluxDB may still be starting..."
fi

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📊 InfluxDB UI:    http://localhost:8086"
echo "   Username: admin"
echo "   Password: factormodel2024"
echo "   Organization: qtab"
echo "   Bucket: jkp_factors"
echo "   Token: qtab_factor_token_2024_secure"
echo ""
echo "📈 Jupyter Lab:    http://localhost:8888"
echo "   Token: qtab_jupyter_2024"
echo ""
echo "🔧 Commands:"
echo "   docker-compose ps    # Check status"
echo "   docker-compose logs  # View logs"
echo "   docker-compose down  # Stop services"
echo ""
echo "📚 Next Steps:"
echo "1. Open InfluxDB UI at http://localhost:8086"
echo "2. Use your existing notebook to load data"
echo "3. Query your data in InfluxDB"
