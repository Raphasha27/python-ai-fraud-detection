#!/bin/bash
set -e

echo "Starting deployment..."

echo "Training model..."
python scripts/train_model.py

echo "Running tests..."
pytest tests/ -v

echo "Building Docker image..."
docker build -t fraud-detection-api .

echo "Pushing to registry..."
docker tag fraud-detection-api raphasha27/fraud-detection-api:latest
docker push raphasha27/fraud-detection-api:latest

echo "Deploying to Cloudflare Workers..."
wrangler deploy

echo "Deployment complete!"
