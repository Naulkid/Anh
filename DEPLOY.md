# Hướng dẫn Triển khai AIDEOM-VN

Tài liệu này mô tả **5 phương án triển khai** từ đơn giản (chạy local)
đến sản xuất (Docker + Cloud), phù hợp với nhiều môi trường khác nhau.

---

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Phương án A — Chạy local (Phát triển)](#2-phương-án-a--chạy-local)
3. [Phương án B — Docker (Khuyến nghị)](#3-phương-án-b--docker)
4. [Phương án C — Streamlit Community Cloud (Miễn phí)](#4-phương-án-c--streamlit-community-cloud)
5. [Phương án D — Google Colab (Sinh viên)](#5-phương-án-d--google-colab)
6. [Phương án E — VPS/Server tổ chức](#6-phương-án-e--vpsserver-tổ-chức)
7. [Kiểm tra sau triển khai](#7-kiểm-tra-sau-triển-khai)
8. [Xử lý sự cố thường gặp](#8-xử-lý-sự-cố-thường-gặp)

---

## 1. Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 2 GB | 4 GB |
| CPU | 2 core | 4 core |
| Disk | 500 MB | 1 GB |
| GPU | Không cần | Không cần |
| OS | Windows/macOS/Linux | Ubuntu 22.04 LTS |

**Phụ thuộc hệ thống (Linux):**
```bash
# GLPK solver — bắt buộc cho Pyomo
sudo apt-get install -y glpk-utils

# macOS
brew install glpk

# Windows — tải từ: http://winglpk.sourceforge.net/
```

---

## 2. Phương án A — Chạy local

**Phù hợp:** Phát triển, kiểm thử, trình bày nhóm nhỏ.

### Bước 1: Chuẩn bị môi trường

```bash
# Clone hoặc giải nén project
unzip aideom_vn_project.zip
cd aideom_vn

# Tạo môi trường ảo Python
python -m venv venv

# Kích hoạt (Linux/macOS)
source venv/bin/activate

# Kích hoạt (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Nâng cấp pip
python -m pip install --upgrade pip
```

### Bước 2: Cài thư viện

```bash
pip install -r requirements.txt
```

> **Lưu ý macOS Apple Silicon (M1/M2/M3):**
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### Bước 3: Kiểm tra cài đặt

```bash
# Kiểm tra core libraries
python -c "import numpy, pandas, scipy, pulp, cvxpy; print('Core OK')"

# Kiểm tra Pyomo + GLPK
python -c "
import pyomo.environ as pyo
s = pyo.SolverFactory('glpk')
print('GLPK available:', s.available())
"

# Kiểm tra Streamlit
python -c "import streamlit; print('Streamlit', streamlit.__version__)"

# Chạy toàn bộ unit tests
python -m pytest tests/ -v --tb=short
```

### Bước 4: Chạy Pipeline kiểm tra

```bash
# Chạy pipeline đầy đủ (tạo biểu đồ vào outputs/)
python src/pipeline.py
```

Kết quả mong đợi:
```
PIPELINE HOÀN THÀNH
Tổng thời gian: ~2s
M1: ~1.2s  M2: ~0.1s  M3: ~0.3s  M4: ~0.05s  M5: ~0.05s
Số lỗi: 0
```

### Bước 5: Khởi động Dashboard

```bash
streamlit run app.py
```

Dashboard mở tại: **http://localhost:8501**

---

## 3. Phương án B — Docker

**Phù hợp:** Triển khai ổn định, tái lập môi trường, CI/CD.

### Yêu cầu
```bash
# Cài Docker
# Ubuntu:
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# macOS/Windows: tải Docker Desktop từ https://docker.com
docker --version  # Docker version 24.x+
```

### Cách 1: Docker Compose (Đơn giản nhất)

```bash
cd aideom_vn

# Build và chạy
docker compose up -d

# Xem logs
docker compose logs -f

# Dừng
docker compose down
```

Dashboard tại: **http://localhost:8501**

### Cách 2: Docker thuần

```bash
# Build image
docker build -t aideom-vn:latest .

# Chạy container
docker run -d \
  --name aideom_vn \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs \
  --restart unless-stopped \
  aideom-vn:latest

# Kiểm tra container
docker ps
docker logs aideom_vn

# Vào container debug
docker exec -it aideom_vn bash
```

### Cách 3: Docker với nginx reverse proxy

Tạo file `nginx.conf`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://aideom-vn:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Bỏ comment phần nginx trong `docker-compose.yml` rồi:
```bash
docker compose up -d
```

---

## 4. Phương án C — Streamlit Community Cloud

**Phù hợp:** Demo công khai, chi phí = 0, không cần server.

### Bước 1: Chuẩn bị GitHub repo

```bash
# Khởi tạo git
cd aideom_vn
git init
git add .
git commit -m "Initial AIDEOM-VN release"

# Tạo repo trên GitHub (github.com/new)
git remote add origin https://github.com/YOUR_USERNAME/aideom-vn.git
git push -u origin main
```

### Bước 2: Deploy lên Streamlit Cloud

1. Truy cập **https://share.streamlit.io**
2. Đăng nhập bằng GitHub
3. Click **"New app"**
4. Cấu hình:
   - Repository: `YOUR_USERNAME/aideom-vn`
   - Branch: `main`
   - Main file: `app.py`
   - Python version: `3.11`
5. Click **"Deploy"**

> ⚠️ **Lưu ý:** Streamlit Cloud không có GLPK. Thêm vào `packages.txt`:
> ```
> glpk-utils
> ```

Tạo file `packages.txt` trong root project:
```bash
echo "glpk-utils" > packages.txt
git add packages.txt && git commit -m "Add system deps" && git push
```

Dashboard public tại: `https://YOUR_USERNAME-aideom-vn-app-HASH.streamlit.app`

---

## 5. Phương án D — Google Colab

**Phù hợp:** Sinh viên không có máy mạnh, trình bày nhanh.

### Notebook setup

Tạo notebook mới trên Colab và chạy:

```python
# Cell 1: Cài đặt môi trường
!apt-get install -y glpk-utils -q
!pip install pulp cvxpy pyomo pymoo streamlit plotly seaborn -q

# Cell 2: Upload project (hoặc clone từ GitHub)
from google.colab import files
# Upload aideom_vn_project.zip
uploaded = files.upload()

!unzip aideom_vn_project.zip -q
%cd aideom_vn

# Cell 3: Kiểm tra
import sys; sys.path.insert(0,'.')
from src.pipeline import AIDEOMPipeline
pipe = AIDEOMPipeline(n_mc_simulations=2000)
outputs = pipe.run_all()
print("Pipeline OK!")

# Cell 4: Chạy Streamlit qua ngrok
!pip install pyngrok -q
from pyngrok import ngrok
import subprocess, threading, time

def run_streamlit():
    subprocess.run(["streamlit","run","app.py",
                    "--server.port=8501","--server.headless=true"])

t = threading.Thread(target=run_streamlit, daemon=True)
t.start()
time.sleep(5)

# Lấy public URL
public_url = ngrok.connect(8501)
print(f"\nDashboard URL: {public_url}")
print("Mở URL trên để xem Dashboard!")
```

---

## 6. Phương án E — VPS/Server tổ chức

**Phù hợp:** Triển khai cho khoa/trường, nhiều người dùng đồng thời.

### Cấu hình VPS khuyến nghị
- **Provider:** DigitalOcean, Vultr, AWS Lightsail, hoặc Bizfly Cloud (VN)
- **Spec:** 2 vCPU, 4GB RAM, 50GB SSD
- **OS:** Ubuntu 22.04 LTS
- **Chi phí:** ~$12–20/tháng

### Triển khai tự động (Script)

```bash
# ====================================================
# deploy.sh — Triển khai AIDEOM-VN lên VPS Ubuntu 22.04
# Chạy với quyền root: sudo bash deploy.sh
# ====================================================

#!/bin/bash
set -e

echo "=== AIDEOM-VN Auto Deploy ==="

# 1. Cập nhật hệ thống
apt-get update && apt-get upgrade -y

# 2. Cài Docker
curl -fsSL https://get.docker.com | sh
systemctl enable docker && systemctl start docker

# 3. Tạo thư mục ứng dụng
mkdir -p /opt/aideom-vn
cd /opt/aideom-vn

# 4. Upload project (scp hoặc git clone)
# Ví dụ: git clone https://github.com/YOUR_USERNAME/aideom-vn.git .

# 5. Build và chạy
docker compose up -d --build

# 6. Cấu hình firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 8501/tcp
ufw --force enable

# 7. Kiểm tra
docker ps
echo "Deploy thành công! Dashboard: http://$(curl -s ifconfig.me):8501"
```

Chạy script:
```bash
# Upload project lên VPS
scp aideom_vn_project.zip root@YOUR_VPS_IP:/opt/

# SSH vào VPS
ssh root@YOUR_VPS_IP

# Giải nén và deploy
cd /opt && unzip aideom_vn_project.zip
cd aideom_vn && bash deploy.sh
```

### Cấu hình Systemd (chạy background không cần Docker)

```bash
# Cài trực tiếp trên server (không dùng Docker)
apt-get install -y glpk-utils python3.11 python3.11-venv

cd /opt/aideom-vn
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Tạo systemd service
cat > /etc/systemd/system/aideom-vn.service << 'EOF'
[Unit]
Description=AIDEOM-VN Streamlit Dashboard
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/aideom-vn
Environment="PATH=/opt/aideom-vn/venv/bin"
ExecStart=/opt/aideom-vn/venv/bin/streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable aideom-vn
systemctl start aideom-vn
systemctl status aideom-vn
```

---

## 7. Kiểm tra sau triển khai

```bash
# 1. Health check endpoint
curl http://localhost:8501/_stcore/health

# 2. Kiểm tra logs
# Docker:
docker logs aideom_vn --tail 50

# Systemd:
journalctl -u aideom-vn -n 50 --no-pager

# 3. Chạy unit tests trong container
docker exec aideom_vn python -m pytest tests/ -q

# 4. Kiểm tra pipeline
docker exec aideom_vn python src/pipeline.py
```

**Kết quả mong đợi sau deploy thành công:**
```
✓ M1 hoàn thành (~1.2s) | MAPE=6.85%
✓ M2 hoàn thành (~0.1s)
✓ M3 hoàn thành (~0.3s) | Z*=64,196 tỷ VND
✓ M4 hoàn thành (~0.05s) | NetJob=1315k việc
✓ M5 hoàn thành (~0.05s) | P50_GDP2030=16,377
PIPELINE HOÀN THÀNH | Số lỗi: 0
```

---

## 8. Xử lý sự cố thường gặp

### Lỗi: `glpk not available`
```bash
# Ubuntu/Debian
sudo apt-get install -y glpk-utils

# macOS
brew install glpk

# Kiểm tra
which glpsol && glpsol --version
```

### Lỗi: `ModuleNotFoundError: No module named 'src'`
```bash
# Phải chạy từ thư mục gốc aideom_vn/
cd aideom_vn
PYTHONPATH=. python src/pipeline.py

# Hoặc thêm vào script:
export PYTHONPATH=/path/to/aideom_vn
```

### Lỗi: `FileNotFoundError: data/vietnam_macro_2020_2025.csv`
```bash
# Kiểm tra file dữ liệu
ls data/

# File cần có:
# data/vietnam_macro_2020_2025.csv
# data/vietnam_sectors_2024.csv
# data/vietnam_regions_2024.csv
```

### Lỗi: `Infeasible` trong M3
```bash
# Kiểm tra ràng buộc equity trong config.py
# Giảm lambda từ 0.62 xuống 0.55:
python -c "
import sys; sys.path.insert(0,'.')
from src.config import BUDGET_CONSTRAINTS
BUDGET_CONSTRAINTS['equity_lambda'] = 0.55
from src.m3_optimization import BudgetOptimizer
r = BudgetOptimizer().solve_pulp()
print('Status:', r.status)
"
```

### Streamlit port đã bị dùng
```bash
# Đổi port
streamlit run app.py --server.port=8502

# Hoặc kill process cũ
lsof -ti:8501 | xargs kill -9
```

### Docker: không đủ RAM
```bash
# Giảm số mô phỏng Monte Carlo trong src/config.py:
# "n_simulations": 5_000  →  2_000

# Hoặc giới hạn RAM Docker:
docker run --memory="2g" aideom-vn:latest
```

---

## Cấu trúc URL Dashboard

| URL | Mô tả |
|---|---|
| `http://localhost:8501` | Dashboard chính |
| `http://localhost:8501/_stcore/health` | Health check |
| `http://localhost:8501/?scenario=S3` | Deep link kịch bản |

---

## Cập nhật dữ liệu

Khi có số liệu mới từ GSO/NSO, chỉ cần thay thế file CSV:

```bash
# Cập nhật dữ liệu
cp vietnam_macro_2026.csv data/vietnam_macro_2020_2026.csv

# Sửa config.py nếu cần thêm năm
# Khởi động lại app (cache sẽ refresh)
# Docker:
docker restart aideom_vn
```

---

## Liên hệ & Hỗ trợ

- **Tài liệu:** `README.md`
- **Issues:** Tạo issue trên GitHub repo
- **Dữ liệu gốc:** [gso.gov.vn](https://www.gso.gov.vn)
- **Tham chiếu:** Nghị quyết 57-NQ/TW, QĐ 749/QĐ-TTg, QĐ 127/QĐ-TTg

---

*AIDEOM-VN v1.0.0 | MIT License | Viện Quản trị Kinh doanh — Trường ĐH Kinh tế*
