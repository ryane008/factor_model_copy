#!/bin/bash

# QTAB Factor Modeling - Docker Quick Start Script
# ===============================================
# This script helps you quickly start the Docker environment for factor modeling research

set -e  # Exit on any error

echo "🚀 QTAB Factor Modeling - Docker Quick Start"
echo "============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker is running"

# Create shared-data directory if it doesn't exist
if [ ! -d "shared-data" ]; then
    mkdir -p shared-data
    echo "📁 Created shared-data directory"
fi

# Check if we have any data files
data_files=$(find shared-data -name "*.csv" -o -name "*.parquet" 2>/dev/null | wc -l)
if [ "$data_files" -eq 0 ]; then
    echo "⚠️  No data files found in shared-data/"
    echo "💡 Place your factor characteristics CSV files in the shared-data/ directory"
    echo "📄 Expected format: characteristics.csv with columns like eom, gvkey, be_me, ret_12_1, etc."
fi

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Service Status:"
docker-compose ps

# Wait for InfluxDB to be ready
echo "⏳ Waiting for InfluxDB to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8086/health > /dev/null 2>&1; then
        echo "✅ InfluxDB is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  InfluxDB may still be starting. Check logs with: docker-compose logs influxdb"
    fi
    sleep 2
done

# Display access information
echo ""
echo "🎯 Access Information:"
echo "====================="
echo "📊 Jupyter Lab:    http://localhost:8888"
echo "   Token:          qtab_jupyter_2024"
echo ""
echo "🗄️  InfluxDB UI:    http://localhost:8086"
echo "   Username:       admin"
echo "   Password:       factormodel2024"
echo "   Organization:   qtab"
echo "   Bucket:         jkp_factors"
echo ""

echo "📝 Next Steps:"
echo "=============="
echo "1. Open Jupyter Lab at http://localhost:8888"
echo "2. Navigate to notebooks/docker_data_access.ipynb"
echo "3. Place your data files in the shared-data/ directory"
echo "4. Run the data loading notebook"
echo ""

echo "🔧 Useful Commands:"
echo "=================="
echo "• View logs:        docker-compose logs -f"
echo "• Stop services:    docker-compose down"
echo "• Restart:          docker-compose restart"
echo "• Shell access:     docker-compose exec jupyter bash"
echo ""

# Check if Jupyter is accessible
echo "🔍 Testing Jupyter accessibility..."
if curl -s http://localhost:8888 > /dev/null 2>&1; then
    echo "✅ Jupyter Lab is accessible"
else
    echo "⚠️  Jupyter Lab may still be starting. Wait a moment and try accessing http://localhost:8888"
fi

echo "✨ Setup complete! Happy factor modeling! 🎉"
