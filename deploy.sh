#!/bin/bash

# MedAssist AI RAG Deployment Script
set -e

echo "🏥 MedAssist AI RAG Deployment Script"
echo "======================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and configure your API keys"
    exit 1
fi

# Load environment variables
source .env

# Validate required environment variables
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  Warning: GEMINI_API_KEY not set. App will run in demo mode."
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🔍 Environment Check:"
echo "   APP_NAME: $APP_NAME"
echo "   APP_VERSION: $APP_VERSION"
echo "   GEMINI_MODEL: $GEMINI_MODEL"
echo "   API_KEY_SET: $(if [ -n "$GEMINI_API_KEY" ]; then echo "✅ Yes"; else echo "❌ No"; fi)"

# Create required directories
echo "📁 Creating directories..."
mkdir -p data logs data/vector_store monitoring

# Build and deploy
echo "🐳 Building Docker image..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

# Health check
echo "🩺 Performing health check..."
for i in {1..30}; do
    if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        echo "✅ Application is healthy!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Health check failed!"
        docker-compose logs medassist-rag
        exit 1
    fi
    echo "   Attempt $i/30..."
    sleep 2
done

echo ""
echo "🎉 Deployment Complete!"
echo "======================================"
echo "🌐 Application URL: http://localhost:8501"
echo "📊 Monitoring (optional): http://localhost:3000 (admin/admin123)"
echo "📝 Logs: docker-compose logs -f medassist-rag"
echo ""
echo "🔧 Management Commands:"
echo "   Stop:    docker-compose down"
echo "   Restart: docker-compose restart"
echo "   Logs:    docker-compose logs -f"
echo "   Update:  ./deploy.sh"

