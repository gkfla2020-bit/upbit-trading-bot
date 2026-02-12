#!/bin/bash
# GCP Cloud Shell에서 실행하세요
# https://console.cloud.google.com/ → 우측 상단 Cloud Shell 아이콘 클릭

# 1. VM 생성 (e2-micro: 무료 등급, us-central1)
gcloud compute instances create trading-bot \
  --zone=us-central1-a \
  --machine-type=e2-micro \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=10GB \
  --tags=trading-bot

echo "VM 생성 완료. 30초 대기..."
sleep 30

# 2. SSH 접속 후 환경 세팅 + 봇 실행
gcloud compute ssh trading-bot --zone=us-central1-a --command='
# Python & pip 설치
sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip git

# 코드 클론
git clone https://github.com/gkfla2020-bit/upbit-trading-bot.git
cd upbit-trading-bot

# 패키지 설치
pip3 install -r requirements.txt --break-system-packages

# .env 파일 생성 (아래 값을 본인 키로 교체하세요!)
cat > .env << EOF
CLAUDE_API_KEY=여기에_클로드_API_키
UPBIT_ACCESS_KEY=여기에_업비트_ACCESS_KEY
UPBIT_SECRET_KEY=여기에_업비트_SECRET_KEY
TELEGRAM_BOT_TOKEN=여기에_텔레그램_봇_토큰
TELEGRAM_CHAT_ID=여기에_텔레그램_채팅_ID
EOF

echo ".env 파일을 수정하세요: nano .env"
echo "수정 후 실행: nohup python3 trading_bot.py > /dev/null 2>&1 &"
'
