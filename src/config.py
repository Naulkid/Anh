"""
config.py — Hằng số & Tham số toàn cục AIDEOM-VN
==================================================
Tập trung tất cả magic numbers và cấu hình kịch bản để dễ
bảo trì và thay thế theo quy định chính sách mới.

Tài liệu tham chiếu:
    - Nghị quyết 57-NQ/TW (2024)
    - QĐ 749/QĐ-TTg (2020) — Chương trình CĐS Quốc gia
    - QĐ 127/QĐ-TTg (2021) — Chiến lược AI đến 2030
    - NSO/GSO Vietnam Statistical Yearbook 2025
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# ─────────────────────────────────────────────
# Đường dẫn dự án
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ─────────────────────────────────────────────
# Tham số hàm sản xuất Cobb-Douglas mở rộng
# Y = A * K^α * L^β * D^γ * AI^δ * H^θ
# Nguồn: ước lượng từ dữ liệu Việt Nam 2020-2025
# ─────────────────────────────────────────────
COBB_DOUGLAS_PARAMS = {
    "alpha": 0.33,   # độ co giãn theo vốn vật chất K
    "beta": 0.42,    # độ co giãn theo lao động L
    "gamma": 0.10,   # độ co giãn theo số hóa D
    "delta": 0.08,   # độ co giãn theo năng lực AI
    "theta": 0.07,   # độ co giãn theo vốn nhân lực số H
}
# Kiểm tra điều kiện CRS: sum = 1.0
assert abs(sum(COBB_DOUGLAS_PARAMS.values()) - 1.0) < 1e-9, \
    "Cobb-Douglas params must sum to 1 (CRS assumption)"

# ─────────────────────────────────────────────
# Điều kiện ban đầu năm 2026 (từ extrapolation 2025)
# ─────────────────────────────────────────────
INITIAL_CONDITIONS_2026 = {
    "K0": 27_500.0,   # nghìn tỷ VND — vốn vật chất
    "L0": 53.9,       # triệu người — lực lượng lao động
    "D0": 20.3,       # % GDP — kinh tế số
    "AI0": 86.0,      # nghìn doanh nghiệp công nghệ số
    "H0": 30.0,       # % — lao động qua đào tạo
    "Y0": 12_847.6,   # nghìn tỷ VND — GDP 2025 (base)
}

# ─────────────────────────────────────────────
# Tham số động học vốn (Bài 8)
# ─────────────────────────────────────────────
CAPITAL_DYNAMICS = {
    "delta_K": 0.05,    # tỷ lệ khấu hao vốn vật chất
    "delta_D": 0.12,    # tỷ lệ khấu hao hạ tầng số
    "delta_AI": 0.15,   # tỷ lệ khấu hao vốn AI
    "theta_H": 0.80,    # hiệu quả chuyển hóa đầu tư GD→nhân lực
    "mu_brain": 0.02,   # chảy máu chất xám
    "phi1": 0.003,      # lan tỏa của D lên TFP
    "phi2": 0.002,      # lan tỏa của AI lên TFP
    "phi3": 0.004,      # lan tỏa của H lên TFP
    "rho": 0.97,        # hệ số chiết khấu liên thời gian
}

# ─────────────────────────────────────────────
# Kịch bản Chính sách (Policy Scenarios)
# K = vốn vật chất, D = số hóa, AI = trí tuệ nhân tạo, H = nhân lực
# ─────────────────────────────────────────────
@dataclass
class ScenarioConfig:
    """Cấu hình phân bổ ngân sách cho một kịch bản chính sách."""
    id: str
    name_vi: str
    name_en: str
    description: str
    allocation: Dict[str, float]   # {"K": w_K, "D": w_D, "AI": w_AI, "H": w_H}
    color: str                     # màu hiển thị trên dashboard
    growth_premium: float = 0.0    # % điều chỉnh tăng trưởng TFP bổ sung

SCENARIOS: Dict[str, ScenarioConfig] = {
    "S1": ScenarioConfig(
        id="S1",
        name_vi="Truyền thống",
        name_en="Baseline Traditional",
        description="Tập trung vốn vật chất, FDI, hạ tầng truyền thống và xuất khẩu",
        allocation={"K": 0.70, "D": 0.10, "AI": 0.10, "H": 0.10},
        color="#6B7280",
        growth_premium=0.0,
    ),
    "S2": ScenarioConfig(
        id="S2",
        name_vi="Số hóa nhanh",
        name_en="Digital Leapfrog",
        description="Tăng đầu tư chính phủ số, doanh nghiệp số, thanh toán số",
        allocation={"K": 0.25, "D": 0.45, "AI": 0.15, "H": 0.15},
        color="#3B82F6",
        growth_premium=0.3,
    ),
    "S3": ScenarioConfig(
        id="S3",
        name_vi="AI dẫn dắt",
        name_en="AI-Led Growth",
        description="Ưu tiên AI, dữ liệu lớn, bán dẫn và trung tâm dữ liệu",
        allocation={"K": 0.20, "D": 0.20, "AI": 0.45, "H": 0.15},
        color="#8B5CF6",
        growth_premium=0.5,
    ),
    "S4": ScenarioConfig(
        id="S4",
        name_vi="Bao trùm số",
        name_en="Inclusive Digital",
        description="Ưu tiên vùng yếu, SME, giáo dục số và nông nghiệp số",
        allocation={"K": 0.30, "D": 0.20, "AI": 0.10, "H": 0.40},
        color="#10B981",
        growth_premium=0.1,
    ),
    "S5": ScenarioConfig(
        id="S5",
        name_vi="Cân bằng tối ưu",
        name_en="Optimal Balanced",
        description="Phân bổ tối ưu theo kết quả mô hình AIDEOM-VN (Pyomo LP)",
        allocation={"K": 0.40, "D": 0.25, "AI": 0.15, "H": 0.20},
        color="#F59E0B",
        growth_premium=0.2,
    ),
}

# ─────────────────────────────────────────────
# Tham số vùng kinh tế cho tối ưu phân bổ (M3)
# ─────────────────────────────────────────────
REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_NAMES_VI = {
    "NMM": "Trung du miền núi phía Bắc",
    "RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ & DH Trung Bộ",
    "CH":  "Tây Nguyên",
    "SE":  "Đông Nam Bộ",
    "MD":  "Đồng bằng sông Cửu Long",
}
INVEST_ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES_VI = {
    "I": "Hạ tầng số",
    "D": "Chuyển đổi số DN",
    "AI": "Năng lực AI",
    "H": "Nhân lực số",
}

# Hệ số tác động biên β_{j,r} — GDP gain (tỷ VND) / tỷ VND đầu tư
BETA_MATRIX = {
    ("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
    ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
    ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
    ("CH",  "I"): 1.20, ("CH",  "D"): 0.75, ("CH",  "AI"): 0.45, ("CH",  "H"): 1.35,
    ("SE",  "I"): 0.90, ("SE",  "D"): 1.30, ("SE",  "AI"): 1.55, ("SE",  "H"): 1.00,
    ("MD",  "I"): 1.10, ("MD",  "D"): 0.85, ("MD",  "AI"): 0.65, ("MD",  "H"): 1.25,
}

# Chỉ số số hóa ban đầu (Bài 4)
DIGITAL_INDEX_INITIAL = {
    "NMM": 38, "RRD": 78, "NCC": 55, "CH": 32, "SE": 82, "MD": 48
}

# ─────────────────────────────────────────────
# Ngưỡng chính sách & ràng buộc ngân sách
# ─────────────────────────────────────────────
BUDGET_CONSTRAINTS = {
    "total_budget_trillion": 50_000,     # ngân sách tổng (tỷ VND)
    "min_per_region": 5_000,             # sàn mỗi vùng
    "max_per_region": 12_000,            # trần mỗi vùng
    "min_human_capital_pct": 0.24,       # ≥24% cho nhân lực số
    "equity_lambda": 0.62,               # hệ số công bằng vùng (λ)
    "equity_gamma": 0.004,               # tác động đầu tư D lên digital index
}

# ─────────────────────────────────────────────
# Tham số Monte Carlo (M5)
# ─────────────────────────────────────────────
MONTE_CARLO_CONFIG = {
    "n_simulations": 10_000,
    "random_seed": 42,
    "gdp_shock_std": 0.025,      # độ lệch chuẩn cú sốc GDP (2.5%)
    "fdi_shock_std": 0.15,       # độ lệch chuẩn FDI (15%)
    "tech_disruption_prob": 0.05,# xác suất gián đoạn công nghệ
}

# ─────────────────────────────────────────────
# Tham số thị trường lao động (M4)
# ─────────────────────────────────────────────
LABOR_PARAMS = {
    # Chỉ số 8 ngành: risk, a1(AI job/tỷ), b1(upgrade/tỷ), c1(displaced/tỷ), d1(retrain/tỷ)
    "sectors": [
        {"name": "Nông-Lâm-Thủy sản", "labor_M": 13.20, "risk": 0.18,
         "a1": 8.5,  "b1": 45.0, "c1": 5.2,  "d1": 50.0},
        {"name": "CN chế biến chế tạo", "labor_M": 11.50, "risk": 0.42,
         "a1": 32.5, "b1": 28.0, "c1": 62.4, "d1": 32.0},
        {"name": "Xây dựng",            "labor_M": 4.80,  "risk": 0.25,
         "a1": 12.8, "b1": 35.0, "c1": 18.5, "d1": 42.0},
        {"name": "Bán buôn-bán lẻ",     "labor_M": 7.80,  "risk": 0.38,
         "a1": 22.4, "b1": 32.0, "c1": 48.2, "d1": 38.0},
        {"name": "Tài chính-Ngân hàng", "labor_M": 0.55,  "risk": 0.52,
         "a1": 45.8, "b1": 22.0, "c1": 72.5, "d1": 26.0},
        {"name": "Logistics-Vận tải",   "labor_M": 1.95,  "risk": 0.35,
         "a1": 28.5, "b1": 30.0, "c1": 42.8, "d1": 36.0},
        {"name": "CNTT-Truyền thông",   "labor_M": 0.62,  "risk": 0.28,
         "a1": 62.5, "b1": 20.0, "c1": 32.5, "d1": 24.0},
        {"name": "Giáo dục-Đào tạo",   "labor_M": 2.15,  "risk": 0.22,
         "a1": 18.5, "b1": 55.0, "c1": 12.5, "d1": 62.0},
    ],
    "total_budget_trillion": 30_000,
}

# ─────────────────────────────────────────────
# Mục tiêu KPI năm 2030 (từ các văn kiện chính sách)
# ─────────────────────────────────────────────
KPI_TARGETS_2030 = {
    "gdp_growth_avg_pct": 7.0,          # tăng trưởng GDP bình quân %/năm
    "digital_economy_pct_gdp": 30.0,    # kinh tế số / GDP (QĐ 749)
    "ai_firms_thousand": 100.0,         # nghìn doanh nghiệp công nghệ số
    "trained_labor_pct": 35.0,          # % lao động qua đào tạo
    "gii_rank_target": 35,              # xếp hạng GII mục tiêu
    "net_zero_year": 2050,              # cam kết COP26
}
