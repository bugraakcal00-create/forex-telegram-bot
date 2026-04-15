#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Forex Bot — Hetzner VPS Kurulum Scripti
#  Kullanım: ssh root@SUNUCU_IP < deploy.sh
# ═══════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════"
echo "  Forex Bot VPS Kurulumu Başlıyor..."
echo "═══════════════════════════════════════════"

# 1. Sistem güncelle
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git nginx certbot python3-certbot-nginx ufw

# 2. Firewall
ufw allow 22
ufw allow 80
ufw allow 443
ufw --force enable

# 3. Bot kullanıcısı oluştur
useradd -m -s /bin/bash forexbot || true
mkdir -p /home/forexbot/app
cd /home/forexbot/app

# 4. Proje dosyalarını kopyala (scp ile gönderilecek)
echo "Proje dosyaları /home/forexbot/app dizinine kopyalanmalı."
echo "Aşağıdaki komutu kendi bilgisayarınızdan çalıştırın:"
echo ""
echo "  scp -r ./* root@SUNUCU_IP:/home/forexbot/app/"
echo ""

# 5. Python ortamı
python3 -m venv /home/forexbot/app/venv
source /home/forexbot/app/venv/bin/activate
pip install --upgrade pip
pip install -r /home/forexbot/app/requirements.txt

# 6. Systemd servisleri
cat > /etc/systemd/system/forexbot-telegram.service << 'UNIT'
[Unit]
Description=ForexBot Telegram Service
After=network.target

[Service]
Type=simple
User=forexbot
WorkingDirectory=/home/forexbot/app
Environment=PYTHONIOENCODING=utf-8
ExecStart=/home/forexbot/app/venv/bin/python -m app.bot
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/forexbot-web.service << 'UNIT'
[Unit]
Description=ForexBot Web Panel
After=network.target

[Service]
Type=simple
User=forexbot
WorkingDirectory=/home/forexbot/app
Environment=PYTHONIOENCODING=utf-8
ExecStart=/home/forexbot/app/venv/bin/uvicorn app.web.server:app --host 127.0.0.1 --port 8081
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

# 7. Nginx reverse proxy
cat > /etc/nginx/sites-available/forexbot << 'NGINX'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8081;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }

    location /static/ {
        alias /home/forexbot/app/app/web/static/;
        expires 7d;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/forexbot /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

# 8. Servisleri başlat
chown -R forexbot:forexbot /home/forexbot/app
systemctl daemon-reload
systemctl enable forexbot-telegram forexbot-web
systemctl start forexbot-telegram forexbot-web

echo ""
echo "═══════════════════════════════════════════"
echo "  KURULUM TAMAMLANDI!"
echo "═══════════════════════════════════════════"
echo ""
echo "  Telegram Bot: systemctl status forexbot-telegram"
echo "  Web Panel:    systemctl status forexbot-web"
echo "  Web Adres:    http://SUNUCU_IP"
echo ""
echo "  SSL eklemek için:"
echo "  certbot --nginx -d senin-domain.com"
echo ""
