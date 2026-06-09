# AIDEOM-VN 🇻🇳
**AI-Driven Economic Optimization Model for Vietnam**

> Mô hình tối ưu hóa kinh tế dựa trên AI cho Việt Nam trong kỷ nguyên chuyển đổi số (2026–2035)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Tổng quan

AIDEOM-VN là hệ thống hỗ trợ ra quyết định (Decision Support System) tích hợp 6 module phân tích kinh tế lượng, tối ưu hóa chính sách và học máy, phục vụ hoạch định chiến lược phát triển kinh tế số Việt Nam giai đoạn 2026–2035.

Mô hình được xây dựng dựa trên bài báo nghiên cứu **"Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI"** và dữ liệu thực tế từ Tổng cục Thống kê (NSO/GSO), Ngân hàng Thế giới, Bộ KH&CN.

### Kiến trúc 6 Module

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  M1: Dự báo │────▶│ M2: Sẵn     │────▶│ M3: Tối ưu  │
│  Kinh tế    │     │ sàng AI/Số  │     │ Phân bổ Vốn │
│ Cobb-Douglas│     │   TOPSIS    │     │    Pyomo    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐     ┌──────▼──────┐
│ M6: Dashboard│◀───│  M5: Rủi ro │◀────│ M4: Lao động│
│  Streamlit  │     │Monte Carlo  │     │ Mô phỏng AI │
│  4 Tab UI   │     │ Stress-test │     │  Thị trường │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 5 Kịch bản Chính sách

| Kịch bản | Mô tả | K | D | AI | H |
|----------|-------|---|---|----|---|
| **S1** Truyền thống | Tập trung vốn vật chất | 70% | 10% | 10% | 10% |
| **S2** Số hóa nhanh | Đẩy mạnh chuyển đổi số | 25% | 45% | 15% | 15% |
| **S3** AI dẫn dắt | Ưu tiên AI & bán dẫn | 20% | 20% | 45% | 15% |
| **S4** Bao trùm số | Phát triển đồng đều | 30% | 20% | 10% | 40% |
| **S5** Cân bằng tối ưu | Kết quả từ mô hình | *Tự động tính* | | | |

---

## 🚀 Cài đặt & Vận hành

### Yêu cầu hệ thống
- Python 3.10 hoặc 3.11
- RAM tối thiểu 4GB
- Không cần GPU (Bài 1–10); GPU tùy chọn (Bài 11–12)

### Bước 1: Clone & Tạo môi trường ảo

```bash
git clone https://github.com/yourname/aideom-vn.git
cd aideom-vn

# Tạo và kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\Activate.ps1   # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 2: Kiểm tra cài đặt

```bash
python -c "import numpy, pandas, scipy, pulp, cvxpy, pyomo; print('✅ Core OK')"
python -c "import streamlit, plotly; print('✅ Dashboard OK')"
python -m pytest tests/ -v     # Chạy toàn bộ unit test
```

### Bước 3: Chạy Dashboard

```bash
streamlit run app.py
```

Dashboard sẽ mở tại `http://localhost:8501`

### Bước 4: Chạy từng module độc lập

```bash
# Module M1 - Dự báo GDP
python src/m1_forecasting.py

# Module M3 - Tối ưu phân bổ vốn
python src/m3_optimization.py

# Chạy tất cả modules theo pipeline
python src/pipeline.py
```

---

## 📁 Cấu trúc Thư mục

```
aideom_vn/
├── 📂 src/                        # Mã nguồn 6 module
│   ├── __init__.py
│   ├── m1_forecasting.py          # M1: Hàm sản xuất Cobb-Douglas + dự báo
│   ├── m2_readiness.py            # M2: Chỉ số sẵn sàng AI/Số (TOPSIS)
│   ├── m3_optimization.py         # M3: Tối ưu phân bổ vốn (Pyomo/PuLP)
│   ├── m4_labor.py                # M4: Mô phỏng tác động lao động
│   ├── m5_risk.py                 # M5: Phân tích rủi ro Monte Carlo
│   ├── m6_dashboard.py            # M6: Logic nghiệp vụ Dashboard
│   ├── data_loader.py             # Tiện ích nạp & kiểm tra dữ liệu
│   ├── config.py                  # Hằng số & tham số toàn cục
│   └── pipeline.py                # Orchestrator chạy toàn bộ pipeline
│
├── 📂 data/                       # Dữ liệu kinh tế Việt Nam
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_sectors_2024.csv
│   └── vietnam_regions_2024.csv
│
├── 📂 tests/                      # Unit tests (pytest)
│   ├── __init__.py
│   ├── test_m1_forecasting.py
│   ├── test_m3_optimization.py
│   └── test_utils.py
│
├── 📂 outputs/                    # Kết quả: bảng, biểu đồ, CSV
├── 📂 reports/                    # Báo cáo Word/PDF
├── app.py                         # Entry point Streamlit Dashboard
├── requirements.txt
└── README.md
```

---

## 📊 Dữ liệu

Toàn bộ bài tập sử dụng dữ liệu thực tế Việt Nam 2020–2025:

| Tệp | Phạm vi | Mô tả |
|-----|---------|-------|
| `vietnam_macro_2020_2025.csv` | Vĩ mô 2020–2025 | GDP, vốn, lao động, số hóa, AI |
| `vietnam_sectors_2024.csv` | 10 ngành 2024 | Tăng trưởng, năng suất, AI Readiness |
| `vietnam_regions_2024.csv` | 6 vùng KT-XH | GRDP, FDI, Digital Index, Gini |

> ⚠️ Nguồn: NSO/GSO, World Bank, Bộ KH&CN, Bộ TT&TT. Số liệu được làm tròn phục vụ giảng dạy.

---

## 🔬 Phương pháp Khoa học

| Module | Phương pháp | Thư viện |
|--------|-------------|---------|
| M1 | Hàm Cobb-Douglas mở rộng, TFP, Growth Accounting | `numpy`, `scipy` |
| M2 | TOPSIS, Entropy Weight, Min-Max normalization | `numpy`, `pandas` |
| M3 | Linear Programming, Stochastic Optimization | `pyomo`, `pulp`, `cvxpy` |
| M4 | NetJob simulation, Markov Chain lao động | `numpy`, `scipy` |
| M5 | Monte Carlo (10,000 simulations), Stress Testing | `numpy`, `scipy` |
| M6 | Interactive Dashboard, Scenario comparison | `streamlit`, `plotly` |

---

## 🧪 Testing

```bash
# Chạy toàn bộ test suite
pytest tests/ -v --tb=short

# Chạy test theo module
pytest tests/test_m1_forecasting.py -v
pytest tests/test_m3_optimization.py -v

# Kiểm tra coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📖 Tài liệu tham khảo

1. Solow, R.M. (1956). "A Contribution to the Theory of Economic Growth"
2. Deb et al. (2002). "NSGA-II: A Fast and Elitist MOEA" — *IEEE TEC*
3. Birge & Louveaux (2011). *Introduction to Stochastic Programming*
4. Tổng cục Thống kê (2026). *Niên giám Thống kê Việt Nam 2025*
5. Nghị quyết 57-NQ/TW (2024) về KH&CN, ĐMST và CĐS quốc gia
6. Quyết định 749/QĐ-TTg (2020) — Chương trình CĐS Quốc gia

---

## 📜 Giấy phép & Liêm chính học thuật

- Mã nguồn: MIT License
- Sinh viên **phải khai báo** việc sử dụng AI hỗ trợ (ChatGPT, Claude, Copilot...) trong báo cáo
- Số liệu học thuật cần truy xuất từ [gso.gov.vn](https://www.gso.gov.vn) và Tổng cục Hải quan

---

*Được phát triển trong khuôn khổ môn học "Mô hình Ra quyết định" — Viện Quản trị Kinh doanh, Trường ĐH Kinh tế*
