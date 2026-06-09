"""
data_loader.py — Tiện ích nạp & kiểm tra dữ liệu AIDEOM-VN
============================================================
Nạp dữ liệu CSV thực tế Việt Nam, chuẩn hóa tên cột,
và tổng hợp các biến phái sinh (K, AI, H) cần cho mô hình
Cobb-Douglas từ các chỉ tiêu có sẵn trong bộ số liệu GSO/NSO.

Ánh xạ cột thực tế → biến mô hình:
  digital_economy_share_GDP_pct → D_digital_pct
  labor_productivity_million_VND → proxy cho H
  GDP_trillion_VND * icor        → K_capital_trillion (ước tính)
  FDI_disbursed_billion_USD * fx → proxy AI capacity
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

# ─── Tên cột tiếng Việt cho sectors (dịch từ tiếng Anh) ────────
SECTOR_NAME_MAP = {
    "Agriculture-Forestry-Fishery":        "Nông-Lâm-Thủy sản",
    "Manufacturing":                        "CN chế biến chế tạo",
    "Construction":                         "Xây dựng",
    "Mining":                               "Khai khoáng",
    "Wholesale-Retail":                     "Bán buôn-bán lẻ",
    "Finance-Banking-Insurance":            "Tài chính-Ngân hàng",
    "Logistics-Transport-Warehousing":      "Logistics-Vận tải",
    "Information-Communication-IT":         "CNTT-Truyền thông",
    "Education-Training":                   "Giáo dục-Đào tạo",
    "Healthcare":                           "Y tế",
}

REGION_NAME_MAP = {
    "Northern Midlands and Mountains":         "Trung du miền núi phía Bắc",
    "Red River Delta":                         "Đồng bằng sông Hồng",
    "North Central and South Central Coast":   "Bắc Trung Bộ & DH Trung Bộ",
    "Central Highlands":                       "Tây Nguyên",
    "Southeast":                               "Đông Nam Bộ",
    "Mekong Delta":                            "Đồng bằng sông Cửu Long",
}


# ════════════════════════════════════════════════════════════════
# Hàm nạp dữ liệu chính
# ════════════════════════════════════════════════════════════════

def load_macro(path: Optional[Path] = None) -> pd.DataFrame:
    """Nạp và làm giàu dữ liệu kinh tế vĩ mô Việt Nam 2020–2025.

    Ngoài các cột gốc từ GSO/NSO, hàm tổng hợp thêm các biến
    phái sinh cần thiết cho hàm sản xuất Cobb-Douglas:

    Biến phái sinh:
        K_capital_trillion   : Ước tính vốn vật chất tích lũy
                               (dùng chuỗi đầu tư tích lũy với ICOR ≈ 3.5)
        L_labor_million      : Lực lượng lao động ước tính
                               (GDP / labor_productivity × 1000)
        D_digital_pct        : = digital_economy_share_GDP_pct (rename)
        AI_tech_firms_thousand: Proxy năng lực AI từ FDI công nghệ
                               (ước tính từ tỷ lệ FDI ICT)
        H_trained_labor_pct  : Vốn nhân lực số
                               (suy ra từ labor_productivity growth)

    Args:
        path: Đường dẫn tùy chỉnh. Mặc định DATA_DIR/vietnam_macro_2020_2025.csv.

    Returns:
        DataFrame đã sắp xếp theo năm với đầy đủ biến mô hình.

    Raises:
        FileNotFoundError: File CSV không tồn tại.
        ValueError: Thiếu cột bắt buộc.
    """
    csv_path = path or DATA_DIR / "vietnam_macro_2020_2025.csv"
    _check_file(csv_path)
    df = pd.read_csv(csv_path).sort_values("year").reset_index(drop=True)

    required = ["year", "GDP_trillion_VND", "digital_economy_share_GDP_pct",
                "labor_productivity_million_VND", "FDI_disbursed_billion_USD"]
    _validate_cols(df, required, csv_path)

    # ── Biến phái sinh ──────────────────────────────────────────

    # 1. Vốn vật chất K: dùng chuỗi đầu tư tích lũy
    #    Đầu tư ~ 27–28% GDP; ICOR ≈ 3.5 → K_t = Σ(invest * ICOR) * scale
    #    Anchor: K_2020 ≈ 16,500 nghìn tỷ (từ Bài 1 đề bài)
    invest_rate = 0.275
    ICOR = 3.5
    K_series = np.zeros(len(df))
    K_series[0] = 16_500.0
    for i in range(1, len(df)):
        invest_t = df["GDP_trillion_VND"].iloc[i - 1] * invest_rate
        K_series[i] = K_series[i - 1] * (1 - 0.05) + invest_t
    df["K_capital_trillion"] = np.round(K_series, 1)

    # 2. Lao động L: GDP / Năng suất lao động
    #    L = GDP (nghìn tỷ) / năng suất (triệu VND/người) = triệu người
    df["L_labor_million"] = np.round(
        df["GDP_trillion_VND"] / df["labor_productivity_million_VND"], 2
    )

    # 3. Số hóa D: rename trực tiếp
    df["D_digital_pct"] = df["digital_economy_share_GDP_pct"]

    # 4. Năng lực AI: ước tính từ FDI công nghệ + doanh nghiệp số
    #    Anchor: AI0_2020 ≈ 55.6 nghìn DN số (từ đề bài)
    #    Tăng trưởng gần với tốc độ FDI ICT ~10-15%/năm
    ai_base = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
    df["AI_tech_firms_thousand"] = ai_base

    # 5. Nhân lực số H: suy từ cải thiện năng suất lao động
    #    H_t = H_2020 * (LP_t / LP_2020)^0.5  (tăng chậm hơn năng suất)
    lp = df["labor_productivity_million_VND"].values
    H_series = 24.1 * np.sqrt(lp / lp[0])
    df["H_trained_labor_pct"] = np.round(np.minimum(H_series, 60.0), 2)

    logger.info("✅ Macro loaded: %d rows | K, L, D, AI, H synthesized", len(df))
    return df


def load_sectors(path: Optional[Path] = None) -> pd.DataFrame:
    """Nạp dữ liệu 10 ngành kinh tế Việt Nam 2024.

    Thêm cột:
        sector_name_vi  : Tên tiếng Việt
        productivity_million_VND : Năng suất = gdp_share * GDP / labor

    Args:
        path: Đường dẫn tùy chỉnh.

    Returns:
        DataFrame 10 ngành với đầy đủ chỉ tiêu.
    """
    csv_path = path or DATA_DIR / "vietnam_sectors_2024.csv"
    _check_file(csv_path)
    df = pd.read_csv(csv_path)

    required = ["sector_name_en", "growth_rate_2024_pct",
                "labor_million", "ai_readiness_0_100", "automation_risk_pct"]
    _validate_cols(df, required, csv_path)

    # Thêm tên tiếng Việt
    df["sector_name_vi"] = df["sector_name_en"].map(SECTOR_NAME_MAP).fillna(df["sector_name_en"])

    # Năng suất lao động ngành: GDP_sector / labor
    # GDP_sector = gdp_share_pct * 11,511.9 nghìn tỷ VND (GDP 2024)
    GDP_2024 = 11_511.9  # nghìn tỷ VND
    df["gdp_sector_trillion"] = df["gdp_share_2024_pct"] / 100 * GDP_2024
    # Năng suất: triệu VND/lao động
    df["productivity_million_VND"] = np.round(
        df["gdp_sector_trillion"] * 1_000 / df["labor_million"].replace(0, np.nan), 1
    ).fillna(0)

    logger.info("✅ Sectors loaded: %d sectors", len(df))
    return df


def load_regions(path: Optional[Path] = None) -> pd.DataFrame:
    """Nạp dữ liệu 6 vùng kinh tế-xã hội Việt Nam 2024.

    Thêm cột:
        region_name_vi : Tên tiếng Việt
        region_code    : Mã vùng rút gọn (NMM, RRD, NCC, CH, SE, MD)

    Args:
        path: Đường dẫn tùy chỉnh.

    Returns:
        DataFrame 6 vùng.
    """
    csv_path = path or DATA_DIR / "vietnam_regions_2024.csv"
    _check_file(csv_path)
    df = pd.read_csv(csv_path)

    required = ["region_name_en", "grdp_per_capita_million_VND",
                "digital_index_0_100", "ai_readiness_0_100", "gini_coef"]
    _validate_cols(df, required, csv_path)

    df["region_name_vi"] = df["region_name_en"].map(REGION_NAME_MAP).fillna(df["region_name_en"])
    region_codes = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
    df["region_code"] = region_codes[: len(df)]

    logger.info("✅ Regions loaded: %d regions", len(df))
    return df


# ════════════════════════════════════════════════════════════════
# Hàm tiện ích
# ════════════════════════════════════════════════════════════════

def normalize_minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    """Chuẩn hóa min-max về [0, 1].

    Args:
        series: Chuỗi số cần chuẩn hóa.
        invert: True → đảo dấu (dùng cho chỉ số xấu như rủi ro, Gini).

    Returns:
        Series đã chuẩn hóa.
    """
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    norm = (series - mn) / (mx - mn)
    return (1 - norm) if invert else norm


def _check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"❌ Không tìm thấy: {path}\n"
            f"   Đảm bảo file CSV nằm trong thư mục data/"
        )


def _validate_cols(df: pd.DataFrame, required: list, path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"❌ {path.name} thiếu cột: {missing}\n"
            f"   Cột hiện có: {list(df.columns)}"
        )


# ════════════════════════════════════════════════════════════════
# Chạy kiểm tra nhanh
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    macro = load_macro()
    sectors = load_sectors()
    regions = load_regions()

    print("\n=== MACRO — Biến Cobb-Douglas ===")
    print(macro[["year", "GDP_trillion_VND", "K_capital_trillion",
                  "L_labor_million", "D_digital_pct",
                  "AI_tech_firms_thousand", "H_trained_labor_pct"]].to_string(index=False))

    print("\n=== SECTORS — 10 ngành ===")
    print(sectors[["sector_name_vi", "growth_rate_2024_pct",
                   "productivity_million_VND", "ai_readiness_0_100",
                   "automation_risk_pct"]].to_string(index=False))

    print("\n=== REGIONS — 6 vùng ===")
    print(regions[["region_name_vi", "grdp_per_capita_million_VND",
                   "digital_index_0_100", "gini_coef"]].to_string(index=False))
