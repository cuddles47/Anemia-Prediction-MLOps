#!/bin/bash

# Cập nhật hệ thống
echo "🔄 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Cài đặt Docker nếu chưa có
if ! command -v docker &> /dev/null
then
    echo "🐳 Docker không được cài. Cài đặt Docker..."
    sudo apt install -y ca-certificates curl gnupg lsb-release
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# Đảm bảo Docker đang chạy
echo "🚀 Starting Docker service..."
sudo systemctl enable docker
sudo systemctl start docker

# Tạo Docker network cho MLflow (tuỳ chọn)
echo "🌐 Creating custom network for MLflow (if not exists)..."
sudo docker network create mlflow-network || echo "Network mlflow-network already exists."

# Cài đặt MLflow thông qua Docker Compose
echo "📦 Creating a Docker Compose file for MLflow..."
cat <<EOF > docker-compose.yml
version: '3'
services:
  mlflow:
    image: mlflow/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_HOME=/mlflow
    volumes:
      - mlflow-data:/mlflow
    networks:
      - mlflow-network
    restart: always

volumes:
  mlflow-data:
    driver: local

networks:
  mlflow-network:
    driver: bridge
EOF

# Chạy Docker Compose để cài MLflow
echo "🚀 Running MLflow container..."
sudo docker-compose up -d

# Kiểm tra MLflow status
echo "✅ Checking if MLflow is running..."
sleep 5
curl -s http://localhost:5000 | grep -i "MLflow"

echo "🎉 MLflow đã được cài đặt và đang chạy tại http://localhost:5000"

