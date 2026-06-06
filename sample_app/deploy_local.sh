#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "Building sample application Docker image..."
docker build -t sample_app -t ai-devops-sample-app .

echo "Stopping old container if it exists..."
docker rm -f ai-devops-sample-app || true

echo "Starting new container..."
docker run -d --name ai-devops-sample-app -p 5000:5000 ai-devops-sample-app

echo "Application deployed locally at http://localhost:5000"
