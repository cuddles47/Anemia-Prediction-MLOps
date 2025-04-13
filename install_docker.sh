#!/bin/bash

# Cập nhật hệ thống
echo "🔄 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Gỡ repo Docker sai nếu có (Ubuntu)
echo "🧹 Removing wrong Docker repo if exists..."
sudo rm -f /etc/apt/sources.list.d/docker.list

# Cài đặt các gói cần thiết
echo "📦 Installing required dependencies..."
sudo apt install -y ca-certificates curl gnupg lsb-release

# Thêm Docker GPG key
echo "🔑 Adding Docker GPG key..."
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Thêm Docker repository phù hợp với Debian 12
echo "📂 Adding Docker APT repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  bookworm stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Cập nhật danh sách gói
echo "🔄 Updating package index with Docker repo..."
sudo apt update

# Cài đặt Docker Engine và các plugin
echo "🐳 Installing Docker Engine and related packages..."
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Thêm user hiện tại vào group docker
CURRENT_USER=$(whoami)
echo "👤 Adding user '$CURRENT_USER' to group docker..."
sudo usermod -aG docker "$CURRENT_USER"

# Kiểm tra Docker daemon
echo "🚀 Enabling and starting Docker service..."
sudo systemctl enable docker
sudo systemctl start docker

# Kiểm tra kết quả
echo "✅ Docker installed. Testing with hello-world..."
sudo docker run hello-world

echo "🎉 Done! You may need to log out and back in for group changes to take effect."

