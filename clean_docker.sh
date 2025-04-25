#!/bin/bash

echo "🔻 Dừng tất cả container..."
sudo docker stop $(sudo docker ps -aq) 2>/dev/null

echo "🗑️  Xóa toàn bộ container, image, volume, network..."
sudo docker system prune -a --volumes -f

echo "❌ Gỡ cài đặt Docker..."
sudo apt-get purge -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin docker-compose

echo "🧹 Xóa thư mục dữ liệu Docker..."
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd
sudo rm -rf ~/.docker

echo "🧼 Xóa socket và group nếu còn..."
sudo rm -f /var/run/docker.sock
sudo groupdel docker 2>/dev/null

echo "✅ Docker đã được gỡ và dọn sạch hoàn toàn!"
