#!/bin/bash

set -e

GREEN='\033[0;32m'
NC='\033[0m'

INSTALLER="Anaconda3-2024.10-1-Linux-x86_64.sh"
URL="https://repo.anaconda.com/archive/$INSTALLER"

echo -e "${GREEN}🚀 Đang cài đặt Anaconda cho Linux...${NC}"

cd ~/script || mkdir -p ~/script && cd ~/script

# Xoá bản cũ nếu tồn tại
[ -f "$INSTALLER" ] && rm -f "$INSTALLER"

echo -e "${GREEN}📥 Đang tải $INSTALLER ...${NC}"
wget "$URL"

echo -e "${GREEN}📦 Đang cài đặt Anaconda...${NC}"
bash "$INSTALLER" -b -p "$HOME/anaconda3"

# Thêm vào PATH nếu chưa có
if ! grep -q 'anaconda3/bin' ~/.bashrc; then
  echo -e "${GREEN}🔧 Đang thêm Anaconda vào PATH...${NC}"
  echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
fi

echo -e "${GREEN}✅ Nạp lại bashrc...${NC}"
source ~/.bashrc

echo -e "${GREEN}🔍 Phiên bản Conda:$(conda --version)${NC}"
echo -e "${GREEN}🎉 Hoàn tất cài đặt Anaconda!${NC}"

