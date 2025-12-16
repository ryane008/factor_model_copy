#!/bin/bash
# Copy factor-modeling files to backtester repo, excluding unwanted directories

SOURCE_DIR="/Users/ryaneng/Desktop/Systematic/QTAB/factor-modeling"
DEST_DIR="/Users/ryaneng/Desktop/browncs/backtester/ryan"

echo "Moving essential files from factor-modeling to backtester..."

# Use rsync to copy with exclusions - exclude unwanted directories but keep essential files
echo "Copying files with rsync (excluding unwanted directories)..."
rsync -av --progress \
  --exclude='qtab_env/' \
  --exclude='alan/' \
  --exclude='src/' \
  --exclude='.git/' \
  --exclude='.env' \
  --exclude='notebooks/' \
  --exclude='us_equity.csv' \
  "$SOURCE_DIR/" "$DEST_DIR/"

echo "Files copied successfully to $DEST_DIR."