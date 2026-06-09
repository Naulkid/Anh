# ─────────────────────────────────────────────────────────────
# Dockerfile — AIDEOM-VN Dashboard
# ─────────────────────────────────────────────────────────────
# Image gốc: Python 3.11 slim (nhỏ hơn full ~300MB)
FROM python:3.11-slim

# Metadata
LABEL maintainer="AIDEOM-VN Team"
LABEL description="AI-Driven Economic Optimization Model for Vietnam"
LABEL version="1.0.0"

# Biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Cài GLPK solver (cần cho Pyomo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    glpk-utils \
    libglpk-dev \
    && rm -rf /var/lib/apt/lists/*

# Thư mục làm việc
WORKDIR /app

# Sao chép requirements trước (tận dụng Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn
COPY . .

# Tạo thư mục outputs
RUN mkdir -p outputs

# Expose cổng Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Lệnh khởi động
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.maxUploadSize=50"]
