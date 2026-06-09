"""
m1_forecasting.py — Module M1: Dự báo Kinh tế Việt Nam
=========================================================
Triển khai hàm sản xuất Cobb-Douglas mở rộng với các yếu tố AI và số hóa,
bao gồm:
  1. Ước lượng TFP (Total Factor Productivity) từ dữ liệu lịch sử.
  2. Dự báo GDP Việt Nam 2026–2035 theo 5 kịch bản chính sách.
  3. Phân rã tăng trưởng (Growth Accounting) — đóng góp của K, L, D, AI, H, TFP.
  4. Phân tích độ nhạy kịch bản đến 2030.

Mô hình:
    Y_t = A_t · K_t^α · L_t^β · D_t^γ · AI_t^δ · H_t^θ
    với α+β+γ+δ+θ = 1 (CRS — Constant Returns to Scale)

Phân rã tăng trưởng:
    ΔlnY = ΔlnA + α·ΔlnK + β·ΔlnL + γ·ΔlnD + δ·ΔlnAI + θ·ΔlnH

Tài liệu tham khảo:
    Solow (1956), Romer (1990), Brynjolfsson et al. (2021),
    NSO/GSO Vietnam Statistical Yearbook 2025.

Author: AIDEOM-VN Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# ─── import nội bộ ───────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    COBB_DOUGLAS_PARAMS, INITIAL_CONDITIONS_2026,
    CAPITAL_DYNAMICS, SCENARIOS, KPI_TARGETS_2030,
    OUTPUTS_DIR,
)
from src.data_loader import load_macro

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Dataclass kết quả dự báo
# ═══════════════════════════════════════════════════════════════

@dataclass
class ForecastResult:
    """Kết quả đầu ra của Module M1.

    Attributes:
        years: Danh sách năm dự báo.
        gdp: Mảng GDP dự báo (nghìn tỷ VND).
        tfp: Mảng TFP ước lượng/dự báo.
        growth_rates: Tỷ lệ tăng trưởng GDP (%).
        capital: Mảng vốn vật chất K.
        digital: Mảng chỉ số số hóa D.
        ai_cap: Mảng năng lực AI.
        human_cap: Mảng vốn nhân lực H.
        scenario_id: Mã kịch bản ("S1"…"S5").
        mape: Mean Absolute Percentage Error của fit lịch sử (%).
        growth_decomposition: DataFrame phân rã tăng trưởng.
    """
    years: List[int]
    gdp: np.ndarray
    tfp: np.ndarray
    growth_rates: np.ndarray
    capital: np.ndarray
    digital: np.ndarray
    ai_cap: np.ndarray
    human_cap: np.ndarray
    scenario_id: str
    mape: float = 0.0
    growth_decomposition: Optional[pd.DataFrame] = None


# ═══════════════════════════════════════════════════════════════
# Class chính: CobbDouglasModel
# ═══════════════════════════════════════════════════════════════

class CobbDouglasModel:
    """Hàm sản xuất Cobb-Douglas mở rộng cho nền kinh tế Việt Nam.

    Mô hình đưa thêm ba yếu tố số hóa (D), năng lực AI, và vốn
    nhân lực số (H) vào framework sản xuất cổ điển, cho phép định
    lượng đóng góp của chuyển đổi số vào tăng trưởng GDP.

    Args:
        params: Dict hệ số {alpha, beta, gamma, delta, theta}.
                Mặc định lấy từ config.COBB_DOUGLAS_PARAMS.

    Example:
        >>> model = CobbDouglasModel()
        >>> model.fit(df_macro)
        >>> result = model.forecast(2026, 2035, scenario_id="S3")
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or COBB_DOUGLAS_PARAMS.copy()
        self._validate_params()

        self.alpha = self.params["alpha"]
        self.beta  = self.params["beta"]
        self.gamma = self.params["gamma"]
        self.delta = self.params["delta"]
        self.theta = self.params["theta"]

        self.df_hist: Optional[pd.DataFrame] = None
        self.tfp_hist: Optional[np.ndarray] = None
        self.tfp_trend_slope: float = 0.0
        self.tfp_mean: float = 1.0
        self._is_fitted: bool = False

    # ─── Kiểm tra tham số ──────────────────────────────────────

    def _validate_params(self) -> None:
        """Kiểm tra tổng hệ số = 1 (CRS condition)."""
        total = sum(self.params.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Tổng hệ số Cobb-Douglas = {total:.4f} ≠ 1.0 (vi phạm CRS). "
                "Hãy điều chỉnh alpha, beta, gamma, delta, theta."
            )

    # ─── Tính sản lượng ────────────────────────────────────────

    def compute_output(
        self,
        A: float | np.ndarray,
        K: float | np.ndarray,
        L: float | np.ndarray,
        D: float | np.ndarray,
        AI: float | np.ndarray,
        H: float | np.ndarray,
    ) -> np.ndarray:
        """Tính sản lượng Y theo hàm Cobb-Douglas mở rộng.

        Args:
            A: TFP (Total Factor Productivity).
            K: Vốn vật chất (nghìn tỷ VND).
            L: Lao động (triệu người).
            D: Chỉ số số hóa (% GDP kinh tế số).
            AI: Năng lực AI (nghìn doanh nghiệp công nghệ).
            H: Vốn nhân lực số (% lao động qua đào tạo).

        Returns:
            Sản lượng Y (nghìn tỷ VND).

        Example:
            >>> model = CobbDouglasModel()
            >>> y = model.compute_output(A=2.5, K=23500, L=52.9,
            ...                          D=18.3, AI=73.8, H=28.4)
        """
        return (
            A
            * np.power(K,  self.alpha)
            * np.power(L,  self.beta)
            * np.power(D,  self.gamma)
            * np.power(AI, self.delta)
            * np.power(H,  self.theta)
        )

    # ─── Ước lượng TFP từ dữ liệu lịch sử ─────────────────────

    def fit(self, df: Optional[pd.DataFrame] = None) -> "CobbDouglasModel":
        """Ước lượng TFP lịch sử bằng phương pháp giải ngược.

        Với mỗi năm t:  A_t = Y_t / (K_t^α · L_t^β · D_t^γ · AI_t^δ · H_t^θ)

        Sau đó hồi quy tuyến tính xu hướng TFP để làm cơ sở dự báo.

        Args:
            df: DataFrame lịch sử. Nếu None, tự động tải từ data/.

        Returns:
            self (để chain method).
        """
        if df is None:
            df = load_macro()

        self.df_hist = df.copy()

        K   = df["K_capital_trillion"].values.astype(float)
        L   = df["L_labor_million"].values.astype(float)
        D   = df["D_digital_pct"].values.astype(float)
        AI  = df["AI_tech_firms_thousand"].values.astype(float)
        H   = df["H_trained_labor_pct"].values.astype(float)
        Y   = df["GDP_trillion_VND"].values.astype(float)

        # Giải ngược: A_t = Y / (K^α · L^β · D^γ · AI^δ · H^θ)
        denom = (
            np.power(K,  self.alpha)
            * np.power(L,  self.beta)
            * np.power(D,  self.gamma)
            * np.power(AI, self.delta)
            * np.power(H,  self.theta)
        )
        self.tfp_hist = Y / denom
        self.tfp_mean = float(np.mean(self.tfp_hist))

        # Xu hướng TFP theo thời gian (OLS trên log-TFP)
        years = df["year"].values.astype(float)
        log_tfp = np.log(self.tfp_hist)
        slope, intercept, r_value, _, _ = linregress(years, log_tfp)
        self.tfp_trend_slope = slope          # tốc độ tăng TFP log/năm
        self.tfp_intercept   = intercept
        self.tfp_r2          = r_value ** 2

        logger.info(
            "TFP ước lượng: mean=%.4f | xu hướng=%.4f/năm | R²=%.3f",
            self.tfp_mean, self.tfp_trend_slope, self.tfp_r2,
        )

        # Tính MAPE kiểm tra fit
        Y_hat = self.compute_output(self.tfp_mean, K, L, D, AI, H)
        self.mape_fit = float(
            np.mean(np.abs((Y - Y_hat) / Y)) * 100
        )
        logger.info("MAPE fit lịch sử: %.2f%%", self.mape_fit)

        self._is_fitted = True
        return self

    # ─── Dự báo 2026–2035 ──────────────────────────────────────

    def forecast(
        self,
        year_start: int = 2026,
        year_end: int = 2035,
        scenario_id: str = "S5",
        tfp_growth_rate: Optional[float] = None,
        custom_allocation: Optional[Dict] = None,
    ) -> ForecastResult:
        """Dự báo GDP theo kịch bản chính sách.

        Phương pháp:
          1. TFP ngoại suy theo xu hướng OLS + điều chỉnh kịch bản.
          2. K, D, AI, H tăng trưởng theo tỷ lệ phân bổ kịch bản.
          3. L tăng tự nhiên ~0.6%/năm (dự báo dân số GSO 2025).
          4. Y_t tính từ hàm Cobb-Douglas.

        Args:
            year_start: Năm bắt đầu dự báo.
            year_end: Năm kết thúc dự báo.
            scenario_id: Mã kịch bản ("S1"…"S5").
            tfp_growth_rate: Ghi đè tốc độ tăng TFP. Mặc định từ xu hướng OLS.
            custom_allocation: Dict phân bổ tùy chỉnh (ghi đè scenario).

        Returns:
            ForecastResult chứa toàn bộ kết quả dự báo.

        Raises:
            RuntimeError: Nếu chưa gọi fit() trước.
        """
        if not self._is_fitted:
            raise RuntimeError("Hãy gọi model.fit() trước khi forecast().")

        scenario = SCENARIOS[scenario_id]
        alloc = custom_allocation or scenario.allocation
        T = year_end - year_start + 1
        years = list(range(year_start, year_end + 1))

        # ── Điều kiện ban đầu ─────────────────────────────────
        ic = INITIAL_CONDITIONS_2026
        K_arr   = np.zeros(T);  K_arr[0]  = ic["K0"]
        L_arr   = np.zeros(T);  L_arr[0]  = ic["L0"]
        D_arr   = np.zeros(T);  D_arr[0]  = ic["D0"]
        AI_arr  = np.zeros(T);  AI_arr[0] = ic["AI0"]
        H_arr   = np.zeros(T);  H_arr[0]  = ic["H0"]
        Y_arr   = np.zeros(T)
        A_arr   = np.zeros(T)
        gR_arr  = np.zeros(T)   # tăng trưởng %

        # ── Ước tính ngân sách đầu tư hàng năm (tỷ VND) ──────
        # Giả định đầu tư ~27% GDP 2026 (tỷ lệ đầu tư Việt Nam)
        annual_invest_base = ic["Y0"] * 0.27   # nghìn tỷ VND
        annual_invest = annual_invest_base * (1 + 0.05)  # tăng 5%/năm

        # ── Tốc độ tăng TFP ──────────────────────────────────
        base_tfp_growth = tfp_growth_rate if tfp_growth_rate is not None \
            else self.tfp_trend_slope   # log-growth/năm
        tfp_growth = base_tfp_growth + scenario.growth_premium * 0.01

        # ── TFP điểm xuất phát ───────────────────────────────
        last_year = float(self.df_hist["year"].max())
        log_tfp_start = self.tfp_intercept + self.tfp_trend_slope * (year_start - 1)
        A_arr[0] = np.exp(log_tfp_start)

        # ── Vòng lặp mô phỏng ────────────────────────────────
        for t in range(T):
            Y_arr[t] = self.compute_output(
                A_arr[t], K_arr[t], L_arr[t],
                D_arr[t], AI_arr[t], H_arr[t],
            )

            if t < T - 1:
                invest = annual_invest * (1 + 0.05 * t)

                # Vốn vật chất: K_{t+1} = (1-δ_K)·K_t + invest·w_K
                K_arr[t+1] = (
                    (1 - CAPITAL_DYNAMICS["delta_K"]) * K_arr[t]
                    + invest * alloc["K"]
                )
                # Số hóa: D_{t+1} = (1-δ_D)·D_t + invest·w_D / scale
                D_arr[t+1] = min(
                    (1 - CAPITAL_DYNAMICS["delta_D"]) * D_arr[t]
                    + invest * alloc["D"] * 0.0015,  # scale factor D (%)
                    45.0  # trần 45% GDP
                )
                # AI: AI_{t+1} = (1-δ_AI)·AI_t + invest·w_AI / scale
                AI_arr[t+1] = (
                    (1 - CAPITAL_DYNAMICS["delta_AI"]) * AI_arr[t]
                    + invest * alloc["AI"] * 0.004
                )
                # Nhân lực: H_{t+1} = H_t + θ_H·invest·w_H / scale - μ·H_t
                H_arr[t+1] = min(
                    H_arr[t]
                    + CAPITAL_DYNAMICS["theta_H"] * invest * alloc["H"] * 0.0006
                    - CAPITAL_DYNAMICS["mu_brain"] * H_arr[t],
                    60.0  # trần 60%
                )
                # Lao động: tăng tự nhiên ~0.6%/năm
                L_arr[t+1] = L_arr[t] * 1.006

                # TFP tăng theo xu hướng + lan tỏa số hóa
                tfp_spillover = (
                    CAPITAL_DYNAMICS["phi1"] * D_arr[t]
                    + CAPITAL_DYNAMICS["phi2"] * AI_arr[t]
                    + CAPITAL_DYNAMICS["phi3"] * H_arr[t]
                )
                A_arr[t+1] = A_arr[t] * (1 + tfp_growth + tfp_spillover * 0.001)

                if t > 0:
                    gR_arr[t] = (Y_arr[t] / Y_arr[t-1] - 1) * 100

        # Tăng trưởng năm đầu (so với 2025)
        gR_arr[0] = (Y_arr[0] / ic["Y0"] - 1) * 100

        return ForecastResult(
            years=years,
            gdp=Y_arr,
            tfp=A_arr,
            growth_rates=gR_arr,
            capital=K_arr,
            digital=D_arr,
            ai_cap=AI_arr,
            human_cap=H_arr,
            scenario_id=scenario_id,
            mape=self.mape_fit,
        )

    # ─── Phân rã tăng trưởng ───────────────────────────────────

    def growth_accounting(self) -> pd.DataFrame:
        """Phân rã tăng trưởng giai đoạn 2020–2025.

        Phương pháp: Lấy sai phân logarit.
          Δln(Y) = Δln(A) + α·Δln(K) + β·Δln(L) + γ·Δln(D) + δ·Δln(AI) + θ·Δln(H)

        Returns:
            DataFrame với cột: year, delta_Y_pct, contrib_K, contrib_L,
            contrib_D, contrib_AI, contrib_H, contrib_TFP.
            Mỗi giá trị là phần trăm đóng góp vào tăng trưởng GDP.

        Raises:
            RuntimeError: Nếu chưa gọi fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Gọi model.fit() trước khi growth_accounting().")

        df = self.df_hist
        n = len(df)
        records = []

        K   = df["K_capital_trillion"].values.astype(float)
        L   = df["L_labor_million"].values.astype(float)
        D   = df["D_digital_pct"].values.astype(float)
        AI  = df["AI_tech_firms_thousand"].values.astype(float)
        H   = df["H_trained_labor_pct"].values.astype(float)
        Y   = df["GDP_trillion_VND"].values.astype(float)
        A   = self.tfp_hist
        yrs = df["year"].values

        for t in range(1, n):
            dln_Y   = np.log(Y[t] / Y[t-1])
            dln_K   = np.log(K[t] / K[t-1])
            dln_L   = np.log(L[t] / L[t-1])
            dln_D   = np.log(D[t] / D[t-1])
            dln_AI  = np.log(AI[t] / AI[t-1])
            dln_H   = np.log(H[t] / H[t-1])
            dln_A   = np.log(A[t] / A[t-1])

            # Đóng góp tuyệt đối (log-change)
            c_K   = self.alpha * dln_K
            c_L   = self.beta  * dln_L
            c_D   = self.gamma * dln_D
            c_AI  = self.delta * dln_AI
            c_H   = self.theta * dln_H
            c_TFP = dln_A

            # Chuyển sang phần trăm của tổng tăng trưởng
            total = abs(dln_Y) if abs(dln_Y) > 1e-10 else 1.0
            scale = 100.0 / (total / dln_Y) if dln_Y != 0 else 100.0

            records.append({
                "year": yrs[t],
                "gdp_growth_pct": round(dln_Y * 100, 2),
                "contrib_K_pp":   round(c_K   * 100, 2),
                "contrib_L_pp":   round(c_L   * 100, 2),
                "contrib_D_pp":   round(c_D   * 100, 2),
                "contrib_AI_pp":  round(c_AI  * 100, 2),
                "contrib_H_pp":   round(c_H   * 100, 2),
                "contrib_TFP_pp": round(c_TFP * 100, 2),
                # Tỷ trọng % trong tổng tăng trưởng
                "share_K_pct":    round(c_K   / dln_Y * 100, 1) if dln_Y else 0,
                "share_L_pct":    round(c_L   / dln_Y * 100, 1) if dln_Y else 0,
                "share_D_pct":    round(c_D   / dln_Y * 100, 1) if dln_Y else 0,
                "share_AI_pct":   round(c_AI  / dln_Y * 100, 1) if dln_Y else 0,
                "share_H_pct":    round(c_H   / dln_Y * 100, 1) if dln_Y else 0,
                "share_TFP_pct":  round(c_TFP / dln_Y * 100, 1) if dln_Y else 0,
            })

        df_decomp = pd.DataFrame(records)
        logger.info("Growth accounting hoàn thành: %d giai đoạn", len(df_decomp))
        return df_decomp

    # ─── So sánh nhiều kịch bản ────────────────────────────────

    def compare_scenarios(
        self, year_end: int = 2030
    ) -> pd.DataFrame:
        """Chạy và so sánh cả 5 kịch bản chính sách đến năm mục tiêu.

        Args:
            year_end: Năm kết thúc so sánh (mặc định 2030).

        Returns:
            DataFrame tổng hợp KPI theo kịch bản:
            scenario_id, name_vi, gdp_2030, growth_avg, digital_2030,
            ai_firms_2030, human_cap_2030, gdp_gain_vs_S1.
        """
        if not self._is_fitted:
            self.fit()

        records = []
        baseline_gdp = None

        for sid, scenario in SCENARIOS.items():
            result = self.forecast(
                year_start=2026,
                year_end=year_end,
                scenario_id=sid,
            )
            gdp_end   = result.gdp[-1]
            gdp_start = result.gdp[0]
            n_years   = len(result.years)
            avg_growth = ((gdp_end / INITIAL_CONDITIONS_2026["Y0"]) ** (1 / n_years) - 1) * 100

            if sid == "S1":
                baseline_gdp = gdp_end

            records.append({
                "scenario_id":       sid,
                "scenario_name_vi":  scenario.name_vi,
                f"gdp_{year_end}_tn_vnd": round(gdp_end, 1),
                "avg_growth_pct":    round(avg_growth, 2),
                f"digital_{year_end}_pct": round(result.digital[-1], 1),
                f"ai_firms_{year_end}_k":  round(result.ai_cap[-1], 1),
                f"human_cap_{year_end}_pct": round(result.human_cap[-1], 1),
            })

        df_cmp = pd.DataFrame(records)
        # Tính GDP gain so với S1 (Baseline)
        col_gdp = f"gdp_{year_end}_tn_vnd"
        df_cmp["gdp_gain_vs_S1_pct"] = (
            (df_cmp[col_gdp] / df_cmp.loc[df_cmp["scenario_id"] == "S1", col_gdp].values[0]) - 1
        ) * 100
        df_cmp["gdp_gain_vs_S1_pct"] = df_cmp["gdp_gain_vs_S1_pct"].round(2)
        return df_cmp


# ═══════════════════════════════════════════════════════════════
# Hàm Vẽ biểu đồ
# ═══════════════════════════════════════════════════════════════

def plot_tfp_trend(model: CobbDouglasModel, save: bool = True) -> plt.Figure:
    """Vẽ xu hướng TFP lịch sử 2020–2025.

    Args:
        model: CobbDouglasModel đã được fit().
        save: Lưu file PNG vào outputs/ nếu True.

    Returns:
        matplotlib Figure object.
    """
    if not model._is_fitted:
        raise RuntimeError("Fit model trước khi plot.")

    years = model.df_hist["year"].values
    A = model.tfp_hist

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(years, A, "o-", color="#2563EB", lw=2.5, ms=8, label="TFP ước lượng")

    # Xu hướng tuyến tính
    trend = np.exp(model.tfp_intercept + model.tfp_trend_slope * years.astype(float))
    ax.plot(years, trend, "--", color="#DC2626", lw=1.5, label=f"Xu hướng OLS (R²={model.tfp_r2:.3f})")

    ax.set_title("Năng suất Nhân tố Tổng hợp (TFP) Việt Nam 2020–2025", fontsize=13, fontweight="bold")
    ax.set_xlabel("Năm")
    ax.set_ylabel("TFP (A_t)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / "m1_tfp_trend.png", dpi=150)
        logger.info("Đã lưu biểu đồ TFP: outputs/m1_tfp_trend.png")

    return fig


def plot_gdp_forecast(
    results: Dict[str, ForecastResult],
    include_history: bool = True,
    save: bool = True,
) -> plt.Figure:
    """Vẽ biểu đồ dự báo GDP theo 5 kịch bản.

    Args:
        results: Dict {scenario_id: ForecastResult}.
        include_history: Nếu True, vẽ thêm dữ liệu lịch sử 2020–2025.
        save: Lưu file PNG.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if include_history:
        try:
            df_hist = load_macro()
            ax.plot(
                df_hist["year"], df_hist["GDP_trillion_VND"],
                "ko-", lw=2, ms=6, label="Lịch sử 2020–2025", zorder=5
            )
        except Exception:
            pass

    colors = {"S1": "#6B7280", "S2": "#3B82F6", "S3": "#8B5CF6",
              "S4": "#10B981", "S5": "#F59E0B"}

    for sid, res in results.items():
        name = SCENARIOS[sid].name_vi
        ax.plot(
            res.years, res.gdp,
            "-", color=colors.get(sid, "gray"),
            lw=2.2, label=f"{sid}: {name}",
        )

    # Đường mục tiêu 2030
    ax.axvline(2030, color="red", linestyle=":", alpha=0.5, lw=1.5)
    ax.text(2030.1, ax.get_ylim()[0] * 1.02, "Mục tiêu 2030", color="red", fontsize=9)

    ax.set_title("Dự báo GDP Việt Nam 2026–2035 theo 5 Kịch bản Chính sách",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Năm")
    ax.set_ylabel("GDP (nghìn tỷ VND)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / "m1_gdp_forecast_scenarios.png", dpi=150)
        logger.info("Đã lưu biểu đồ GDP: outputs/m1_gdp_forecast_scenarios.png")

    return fig


def plot_growth_decomposition(df_decomp: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Vẽ biểu đồ cột phân rã đóng góp tăng trưởng GDP.

    Args:
        df_decomp: DataFrame từ model.growth_accounting().
        save: Lưu file PNG.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    cols = ["contrib_K_pp", "contrib_L_pp", "contrib_D_pp",
            "contrib_AI_pp", "contrib_H_pp", "contrib_TFP_pp"]
    labels = ["Vốn (K)", "Lao động (L)", "Số hóa (D)",
              "AI capacity", "Nhân lực số (H)", "TFP"]
    palette = ["#1E40AF", "#7C3AED", "#059669", "#D97706", "#DC2626", "#374151"]

    x = np.arange(len(df_decomp))
    bottom = np.zeros(len(df_decomp))

    for col, label, color in zip(cols, labels, palette):
        vals = df_decomp[col].values
        ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += vals

    ax.plot(x, df_decomp["gdp_growth_pct"].values, "ko-", lw=2, ms=5, label="Tổng ΔlnY")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in df_decomp["year"].values])
    ax.set_title("Phân rã Đóng góp Tăng trưởng GDP Việt Nam (pp = percentage point)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Đóng góp (điểm phần trăm)")
    ax.legend(loc="upper left", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / "m1_growth_decomposition.png", dpi=150)

    return fig


# ═══════════════════════════════════════════════════════════════
# Entry point chạy độc lập
# ═══════════════════════════════════════════════════════════════

def run_m1_analysis() -> Dict:
    """Chạy toàn bộ phân tích M1 và trả kết quả cho pipeline.

    Returns:
        Dict chứa: model, tfp_df, decomp_df, forecast_results, comparison_df.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("\n" + "="*60)
    print("  MODULE M1 — Hàm sản xuất Cobb-Douglas mở rộng")
    print("  Dự báo GDP Việt Nam 2026–2035")
    print("="*60)

    # 1. Fit mô hình
    model = CobbDouglasModel()
    model.fit()

    # 2. Hiển thị TFP lịch sử
    df = model.df_hist
    tfp_df = pd.DataFrame({
        "year": df["year"],
        "tfp_A": np.round(model.tfp_hist, 4),
    })
    print("\n📊 TFP ước lượng Việt Nam 2020–2025:")
    print(tfp_df.to_string(index=False))
    print(f"\n   MAPE fit lịch sử: {model.mape_fit:.2f}%")

    # 3. Phân rã tăng trưởng
    decomp_df = model.growth_accounting()
    print("\n📊 Phân rã tăng trưởng GDP 2020–2025:")
    display_cols = ["year", "gdp_growth_pct",
                    "share_K_pct", "share_L_pct", "share_D_pct",
                    "share_AI_pct", "share_H_pct", "share_TFP_pct"]
    print(decomp_df[display_cols].to_string(index=False))

    # 4. Dự báo tất cả kịch bản
    forecast_results = {}
    for sid in SCENARIOS:
        forecast_results[sid] = model.forecast(
            year_start=2026, year_end=2035, scenario_id=sid
        )

    # 5. So sánh kịch bản
    comparison_df = model.compare_scenarios(year_end=2030)
    print("\n📊 So sánh 5 kịch bản đến năm 2030:")
    print(comparison_df.to_string(index=False))

    # Kiểm tra mục tiêu KPI
    kpi_target = KPI_TARGETS_2030["gdp_growth_avg_pct"]
    s5_growth = comparison_df.loc[
        comparison_df["scenario_id"] == "S5", "avg_growth_pct"
    ].values[0]
    status = "✅" if s5_growth >= kpi_target else "⚠️"
    print(f"\n{status} Mục tiêu tăng trưởng 7%/năm: S5 đạt {s5_growth:.2f}%")

    # 6. Vẽ biểu đồ
    plot_tfp_trend(model, save=True)
    plot_gdp_forecast(forecast_results, save=True)
    plot_growth_decomposition(decomp_df, save=True)
    print("\n✅ Biểu đồ đã lưu vào thư mục outputs/")

    return {
        "model": model,
        "tfp_df": tfp_df,
        "decomp_df": decomp_df,
        "forecast_results": forecast_results,
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    run_m1_analysis()
