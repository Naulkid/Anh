"""
m5_risk.py — Module M5: Phân tích Rủi ro & Stress Testing
===========================================================
Mô phỏng Monte Carlo (10,000 lần) để định lượng phân phối GDP
dưới bất định về TFP, FDI, và cú sốc bên ngoài; kết hợp với
stress testing các kịch bản cực đoan theo mô hình two-stage
stochastic (Bài 10 đề bài).

Các rủi ro được mô hình hóa:
  1. Rủi ro TFP — biến động năng suất nhân tố tổng hợp
  2. Rủi ro FDI — dòng vốn FDI không ổn định
  3. Rủi ro công nghệ — gián đoạn AI/chuỗi cung ứng bán dẫn
  4. Rủi ro địa-chính trị — xung đột thương mại, sanctions
  5. Rủi ro khí hậu — thiên tai, chi phí chuyển đổi xanh

Đầu ra:
  - Phân phối GDP 2030 (phân vị 5%, 25%, 50%, 75%, 95%)
  - VaR (Value at Risk) và CVaR (Conditional VaR)
  - Phân tích độ nhạy Sobol (variance decomposition)
  - Dashboard rủi ro cho M6

Author: AIDEOM-VN Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    MONTE_CARLO_CONFIG, INITIAL_CONDITIONS_2026,
    COBB_DOUGLAS_PARAMS, OUTPUTS_DIR, SCENARIOS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Dataclass kết quả
# ═══════════════════════════════════════════════════════════════

@dataclass
class RiskResult:
    """Kết quả phân tích rủi ro Monte Carlo.

    Attributes:
        gdp_simulations: Mảng (N_sim,) GDP 2030 mô phỏng (nghìn tỷ VND).
        percentiles: Dict {5, 25, 50, 75, 95} → giá trị GDP.
        var_95: Value-at-Risk 95% (GDP thấp hơn mức này với 5% xác suất).
        cvar_95: Conditional VaR — GDP kỳ vọng trong tail 5% xấu nhất.
        prob_below_target: Xác suất GDP < mục tiêu chính sách.
        sensitivity: DataFrame đóng góp phương sai mỗi yếu tố rủi ro.
        scenario_id: Kịch bản chính sách tương ứng.
        stress_results: Dict kết quả stress test từng kịch bản cực đoan.
        n_simulations: Số lần mô phỏng.
    """
    gdp_simulations: np.ndarray
    percentiles: Dict[int, float]
    var_95: float
    cvar_95: float
    prob_below_target: float
    sensitivity: pd.DataFrame
    scenario_id: str
    stress_results: Dict[str, float] = field(default_factory=dict)
    n_simulations: int = 10_000


# ═══════════════════════════════════════════════════════════════
# Class chính: RiskAnalyzer
# ═══════════════════════════════════════════════════════════════

class RiskAnalyzer:
    """Phân tích rủi ro tích hợp cho AIDEOM-VN.

    Kết hợp Monte Carlo simulation với hàm sản xuất Cobb-Douglas
    để phân phối GDP 2030 dưới nhiều nguồn bất định.

    Args:
        n_simulations: Số lần mô phỏng MC. Mặc định 10,000.
        random_seed: Hạt giống cho tái lập kết quả.
        target_gdp_2030: Mục tiêu GDP 2030 để tính xác suất rủi ro.

    Example:
        >>> analyzer = RiskAnalyzer(n_simulations=10_000)
        >>> result = analyzer.run(scenario_id="S5")
        >>> analyzer.print_summary(result)
    """

    # Tham số phân phối xác suất mỗi yếu tố rủi ro
    # (mean_shock, std_shock, distribution)
    RISK_FACTORS = {
        "tfp_shock":     {"mean": 0.0,   "std": 0.025, "dist": "normal",
                          "label": "Cú sốc TFP"},
        "fdi_shock":     {"mean": 0.0,   "std": 0.15,  "dist": "normal",
                          "label": "Biến động FDI"},
        "tech_disrupt":  {"mean": -0.02, "std": 0.03,  "dist": "normal",
                          "label": "Gián đoạn Công nghệ"},
        "geopolitical":  {"mean": -0.01, "std": 0.04,  "dist": "normal",
                          "label": "Rủi ro Địa-Chính trị"},
        "climate_risk":  {"mean": -0.008,"std": 0.015, "dist": "normal",
                          "label": "Rủi ro Khí hậu"},
    }

    # Kịch bản stress test cực đoan
    STRESS_SCENARIOS = {
        "covid_like":    {"tfp": -0.08, "fdi": -0.35, "tech": -0.05,
                          "label": "Đại dịch COVID-like (-8% TFP)"},
        "yagi_storm":    {"tfp": -0.02, "fdi": -0.05, "tech": 0.0,
                          "label": "Siêu bão Yagi (-2% TFP)"},
        "trade_war":     {"tfp": -0.03, "fdi": -0.25, "tech": -0.08,
                          "label": "Chiến tranh Thương mại (-25% FDI)"},
        "chip_shortage": {"tfp": -0.01, "fdi": -0.10, "tech": -0.12,
                          "label": "Khủng hoảng Bán dẫn (-12% Tech)"},
        "base_case":     {"tfp": 0.0,   "fdi": 0.0,   "tech": 0.0,
                          "label": "Kịch bản Cơ sở"},
    }

    def __init__(
        self,
        n_simulations: int = 10_000,
        random_seed: int = 42,
        target_gdp_2030: float = 14_000.0,  # nghìn tỷ VND — P10 pessimistic threshold
    ):
        self.N = n_simulations
        self.rng = np.random.default_rng(random_seed)
        self.target = target_gdp_2030
        self.alpha = COBB_DOUGLAS_PARAMS["alpha"]
        self.beta  = COBB_DOUGLAS_PARAMS["beta"]
        self.gamma = COBB_DOUGLAS_PARAMS["gamma"]
        self.delta = COBB_DOUGLAS_PARAMS["delta"]
        self.theta = COBB_DOUGLAS_PARAMS["theta"]

    # ─── Hàm sản xuất dạng vectơ ────────────────────────────────

    def _cobb_douglas_vec(
        self,
        A: np.ndarray,
        K: np.ndarray,
        L: np.ndarray,
        D: np.ndarray,
        AI: np.ndarray,
        H: np.ndarray,
    ) -> np.ndarray:
        """Cobb-Douglas mở rộng, vectơ hóa cho N mô phỏng."""
        return (A * np.power(K, self.alpha) * np.power(L, self.beta)
                * np.power(D, self.gamma) * np.power(AI, self.delta)
                * np.power(H, self.theta))

    # ─── Dự báo GDP 2030 với cú sốc ngẫu nhiên ─────────────────

    def _project_gdp_2030(
        self,
        scenario_id: str,
        tfp_shocks: np.ndarray,
        fdi_shocks: np.ndarray,
        tech_shocks: np.ndarray,
        geo_shocks: np.ndarray,
        climate_shocks: np.ndarray,
    ) -> np.ndarray:
        """Chiếu GDP 2030 cho N mô phỏng theo kịch bản chính sách.

        Args:
            scenario_id: Mã kịch bản ("S1"…"S5").
            *_shocks: Mảng (N,) cú sốc từng yếu tố rủi ro.

        Returns:
            Mảng GDP 2030 (N,) nghìn tỷ VND.
        """
        scenario = SCENARIOS[scenario_id]
        alloc = scenario.allocation
        ic = INITIAL_CONDITIONS_2026

        # Tốc độ tăng trưởng cơ sở mỗi yếu tố (5 năm)
        T = 5  # 2026 → 2030
        annual_invest = ic["Y0"] * 0.275  # nghìn tỷ/năm

        # Tích lũy vốn theo kịch bản (giá trị trung bình 2030)
        K_2030 = ic["K0"] * (1 - 0.05)**T + annual_invest * alloc["K"] * T
        D_2030 = min(ic["D0"] + alloc["D"] * 3.5 * T, 40.0)  # D grows ~0.7pp/yr per pct allocation
        AI_2030 = ic["AI0"] * (1 - 0.15)**T + annual_invest * alloc["AI"] * 0.003 * T
        H_2030 = min(ic["H0"] + alloc["H"] * 0.8 * T, 50.0)
        L_2030 = ic["L0"] * (1.006 ** T)

        # TFP cơ sở 2030 — tính từ dữ liệu 2025 thực tế
        # A_2025 ≈ 35.58 (giải ngược Cobb-Douglas từ GDP 2025)
        A_2025 = 35.58
        tfp_annual_growth = 0.048   # xu hướng OLS từ M1
        A_base = A_2025 * (1 + tfp_annual_growth) ** T * (1 + scenario.growth_premium * 0.01)

        # Áp dụng cú sốc ngẫu nhiên cho N mô phỏng
        A_sim  = A_base  * (1 + tfp_shocks + tech_shocks + geo_shocks + climate_shocks)
        K_sim  = K_2030  * (1 + fdi_shocks * 0.3)   # FDI ảnh hưởng ~30% vốn
        AI_sim = AI_2030 * (1 + fdi_shocks * 0.5 + tech_shocks)

        # Đảm bảo giá trị dương
        A_sim  = np.maximum(A_sim,  0.1)
        K_sim  = np.maximum(K_sim,  ic["K0"] * 0.5)
        AI_sim = np.maximum(AI_sim, 10.0)

        gdp = self._cobb_douglas_vec(
            A_sim, K_sim, L_2030, D_2030, AI_sim, H_2030
        )
        return gdp

    # ─── Monte Carlo chính ──────────────────────────────────────

    def run(
        self,
        scenario_id: str = "S5",
        include_stress: bool = True,
    ) -> RiskResult:
        """Chạy toàn bộ phân tích rủi ro Monte Carlo.

        Args:
            scenario_id: Kịch bản chính sách.
            include_stress: Có chạy stress test không.

        Returns:
            RiskResult với đầy đủ thống kê phân phối.
        """
        logger.info("Đang chạy %d mô phỏng Monte Carlo [%s]...", self.N, scenario_id)

        # ── Sinh cú sốc ngẫu nhiên ─────────────────────────────
        rf = self.RISK_FACTORS
        shocks = {
            "tfp":    self.rng.normal(rf["tfp_shock"]["mean"],
                                      rf["tfp_shock"]["std"], self.N),
            "fdi":    self.rng.normal(rf["fdi_shock"]["mean"],
                                      rf["fdi_shock"]["std"], self.N),
            "tech":   self.rng.normal(rf["tech_disrupt"]["mean"],
                                      rf["tech_disrupt"]["std"], self.N),
            "geo":    self.rng.normal(rf["geopolitical"]["mean"],
                                      rf["geopolitical"]["std"], self.N),
            "climate":self.rng.normal(rf["climate_risk"]["mean"],
                                      rf["climate_risk"]["std"], self.N),
        }

        # ── Chiếu GDP cho N mô phỏng ───────────────────────────
        gdp_sims = self._project_gdp_2030(
            scenario_id,
            shocks["tfp"], shocks["fdi"], shocks["tech"],
            shocks["geo"], shocks["climate"],
        )

        # ── Thống kê phân phối ─────────────────────────────────
        pcts = {p: float(np.percentile(gdp_sims, p))
                for p in [5, 25, 50, 75, 95]}
        var_95  = float(np.percentile(gdp_sims, 5))   # worst 5%
        cvar_95 = float(gdp_sims[gdp_sims <= var_95].mean())
        prob_below = float((gdp_sims < self.target).mean())

        # ── Phân tích độ nhạy (one-at-a-time) ─────────────────
        sensitivity_df = self._sensitivity_analysis(scenario_id, shocks)

        # ── Stress testing ─────────────────────────────────────
        stress_results = {}
        if include_stress:
            stress_results = self._stress_test(scenario_id)

        logger.info(
            "MC hoàn thành: P50=%.0f | VaR95=%.0f | P(below target)=%.1f%%",
            pcts[50], var_95, prob_below * 100
        )

        return RiskResult(
            gdp_simulations=gdp_sims,
            percentiles=pcts,
            var_95=var_95,
            cvar_95=cvar_95,
            prob_below_target=prob_below,
            sensitivity=sensitivity_df,
            scenario_id=scenario_id,
            stress_results=stress_results,
            n_simulations=self.N,
        )

    def _sensitivity_analysis(
        self,
        scenario_id: str,
        base_shocks: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Phân tích độ nhạy: phương sai GDP do mỗi yếu tố.

        Phương pháp OAT (One-At-a-Time): giữ 4 yếu tố = 0,
        chỉ biến thiên 1 yếu tố để đo phương sai.

        Args:
            scenario_id: Mã kịch bản.
            base_shocks: Dict cú sốc đã sinh.

        Returns:
            DataFrame: factor, variance_contribution_pct, std_gdp.
        """
        total_gdp = self._project_gdp_2030(
            scenario_id,
            base_shocks["tfp"], base_shocks["fdi"], base_shocks["tech"],
            base_shocks["geo"], base_shocks["climate"],
        )
        total_var = np.var(total_gdp)

        zeros = np.zeros(self.N)
        records = []
        for factor_name, shocks_arr in base_shocks.items():
            # Chỉ bật yếu tố này, tắt phần còn lại
            kwargs = {k: zeros for k in base_shocks}
            kwargs[factor_name] = shocks_arr
            gdp_oat = self._project_gdp_2030(
                scenario_id,
                kwargs["tfp"], kwargs["fdi"], kwargs["tech"],
                kwargs["geo"], kwargs["climate"],
            )
            var_oat = np.var(gdp_oat)
            label = self.RISK_FACTORS.get(
                factor_name + "_shock",
                self.RISK_FACTORS.get(factor_name, {})
            ).get("label", factor_name)
            # lookup label properly
            label_map = {
                "tfp": "Cú sốc TFP", "fdi": "Biến động FDI",
                "tech": "Gián đoạn Công nghệ",
                "geo": "Rủi ro Địa-Chính trị",
                "climate": "Rủi ro Khí hậu",
            }
            records.append({
                "factor": label_map.get(factor_name, factor_name),
                "variance_gdp": round(var_oat, 2),
                "pct_of_total": round(var_oat / max(total_var, 1e-9) * 100, 1),
                "std_gdp_tn": round(np.std(gdp_oat), 1),
            })

        df = pd.DataFrame(records).sort_values("pct_of_total", ascending=False)
        return df.reset_index(drop=True)

    def _stress_test(self, scenario_id: str) -> Dict[str, float]:
        """Chạy stress test cho các kịch bản cực đoan.

        Args:
            scenario_id: Kịch bản chính sách cơ sở.

        Returns:
            Dict {scenario_name: median_gdp_2030}.
        """
        results = {}
        for stress_id, stress in self.STRESS_SCENARIOS.items():
            tfp_s  = np.full(self.N, stress.get("tfp",  0.0))
            fdi_s  = np.full(self.N, stress.get("fdi",  0.0))
            tech_s = np.full(self.N, stress.get("tech", 0.0))
            gdp_s  = self._project_gdp_2030(
                scenario_id, tfp_s, fdi_s, tech_s,
                np.zeros(self.N), np.zeros(self.N)
            )
            results[stress_id] = float(np.median(gdp_s))
        return results

    # ─── So sánh rủi ro qua 5 kịch bản ─────────────────────────

    def compare_scenario_risks(self) -> pd.DataFrame:
        """Chạy MC cho cả 5 kịch bản và so sánh chỉ số rủi ro.

        Returns:
            DataFrame: scenario, p50_gdp, var95, cvar95,
                       prob_miss_target, risk_premium.
        """
        records = []
        for sid in SCENARIOS:
            r = self.run(scenario_id=sid, include_stress=False)
            records.append({
                "scenario_id":    sid,
                "scenario_name":  SCENARIOS[sid].name_vi,
                "p50_gdp_tn":     round(r.percentiles[50], 0),
                "p5_gdp_tn":      round(r.percentiles[5],  0),
                "p95_gdp_tn":     round(r.percentiles[95], 0),
                "var_95":         round(r.var_95, 0),
                "cvar_95":        round(r.cvar_95, 0),
                "prob_miss_pct":  round(r.prob_below_target * 100, 1),
                "uncertainty_range": round(
                    r.percentiles[95] - r.percentiles[5], 0),
            })
        df = pd.DataFrame(records)
        # Risk premium: gap giữa P50 và CVaR
        df["risk_premium_tn"] = (df["p50_gdp_tn"] - df["cvar_95"]).round(0)
        return df

    def print_summary(self, result: RiskResult) -> None:
        """In tóm tắt rủi ro ra console."""
        p = result.percentiles
        print(f"\n{'='*58}")
        print(f"  M5 — Phân tích Rủi ro [{result.scenario_id}]")
        print(f"  N = {result.n_simulations:,} mô phỏng Monte Carlo")
        print(f"{'='*58}")
        print(f"  Phân phối GDP 2030 (nghìn tỷ VND):")
        print(f"    P5  (xấu nhất 5%): {p[5]:>8,.0f}")
        print(f"    P25:               {p[25]:>8,.0f}")
        print(f"    P50 (trung vị):    {p[50]:>8,.0f}")
        print(f"    P75:               {p[75]:>8,.0f}")
        print(f"    P95 (tốt nhất 5%): {p[95]:>8,.0f}")
        print(f"\n  VaR (95%):  {result.var_95:>8,.0f} nghìn tỷ VND")
        print(f"  CVaR (95%): {result.cvar_95:>8,.0f} nghìn tỷ VND")
        print(f"  P(GDP < {self.target:,.0f}): {result.prob_below_target*100:.1f}%")
        print(f"\n  Đóng góp Phương sai (Sensitivity):")
        print(result.sensitivity[["factor","pct_of_total","std_gdp_tn"]].to_string(index=False))
        if result.stress_results:
            print(f"\n  Stress Test — GDP trung vị 2030:")
            for k, v in result.stress_results.items():
                label = self.STRESS_SCENARIOS[k]["label"]
                print(f"    {label[:38]:<40}: {v:>8,.0f}")
        print("="*58)


# ═══════════════════════════════════════════════════════════════
# Biểu đồ
# ═══════════════════════════════════════════════════════════════

def plot_risk_dashboard(result: RiskResult, save: bool = True) -> plt.Figure:
    """Dashboard 4 ô: phân phối, CDF, sensitivity, stress test.

    Args:
        result: RiskResult từ RiskAnalyzer.run().
        save: Lưu PNG.

    Returns:
        matplotlib Figure.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    gdp = result.gdp_simulations
    p   = result.percentiles
    colors = {"S1":"#6B7280","S2":"#3B82F6","S3":"#8B5CF6","S4":"#10B981","S5":"#F59E0B"}
    c = colors.get(result.scenario_id, "#374151")

    # ── (1) Histogram phân phối GDP ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(gdp, bins=80, color=c, alpha=0.70, edgecolor="white", lw=0.3)
    for pct, val in [(5, p[5]), (50, p[50]), (95, p[95])]:
        ax1.axvline(val, color="red" if pct==5 else ("green" if pct==95 else "navy"),
                    ls="--", lw=1.5, label=f"P{pct}={val:,.0f}")
    ax1.set_title(f"Phân phối GDP 2030 [{result.scenario_id}]\n(N={result.n_simulations:,} mô phỏng)",
                  fontsize=10, fontweight="bold")
    ax1.set_xlabel("GDP 2030 (nghìn tỷ VND)")
    ax1.set_ylabel("Tần suất")
    ax1.legend(fontsize=8)

    # ── (2) CDF ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_gdp = np.sort(gdp)
    cdf = np.arange(1, len(sorted_gdp)+1) / len(sorted_gdp)
    ax2.plot(sorted_gdp, cdf * 100, color=c, lw=2)
    ax2.axhline(5,  color="red",  ls=":", lw=1.2, alpha=0.7, label="VaR 95%")
    ax2.axhline(50, color="navy", ls=":", lw=1.2, alpha=0.7, label="Median")
    ax2.set_title("CDF — Phân vị Tích lũy", fontsize=10, fontweight="bold")
    ax2.set_xlabel("GDP 2030 (nghìn tỷ VND)")
    ax2.set_ylabel("Xác suất tích lũy (%)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

    # ── (3) Sensitivity (variance decomposition) ───────────────
    ax3 = fig.add_subplot(gs[1, 0])
    df_s = result.sensitivity
    bars = ax3.barh(df_s["factor"], df_s["pct_of_total"],
                    color=["#EF4444","#F97316","#EAB308","#22C55E","#3B82F6"][:len(df_s)],
                    alpha=0.85)
    for bar, val in zip(bars, df_s["pct_of_total"]):
        ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9)
    ax3.set_xlabel("Đóng góp Phương sai (%)")
    ax3.set_title("Phân tích Độ nhạy — Nguồn Rủi ro Chính",
                  fontsize=10, fontweight="bold")
    ax3.set_xlim(0, df_s["pct_of_total"].max() * 1.25)
    ax3.grid(axis="x", alpha=0.25)

    # ── (4) Stress Test ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    stress = result.stress_results
    if stress:
        analyzer = RiskAnalyzer.__new__(RiskAnalyzer)
        labels = [RiskAnalyzer.STRESS_SCENARIOS[k]["label"][:30]
                  for k in stress]
        vals = list(stress.values())
        bar_colors = ["#10B981" if v >= p[50]*0.95 else
                      "#F59E0B" if v >= p[50]*0.88 else "#EF4444"
                      for v in vals]
        bars2 = ax4.barh(labels, vals, color=bar_colors, alpha=0.85)
        ax4.axvline(p[50], color="navy", ls="--", lw=1.5, label=f"P50={p[50]:,.0f}")
        for bar, val in zip(bars2, vals):
            ax4.text(val + max(vals)*0.005, bar.get_y() + bar.get_height()/2,
                     f"{val:,.0f}", va="center", fontsize=8)
        ax4.set_xlabel("GDP trung vị 2030 (nghìn tỷ VND)")
        ax4.set_title("Stress Test — Kịch bản Cực đoan",
                      fontsize=10, fontweight="bold")
        ax4.legend(fontsize=8)
        ax4.grid(axis="x", alpha=0.25)

    fig.suptitle(f"Dashboard Rủi ro AIDEOM-VN — Kịch bản {result.scenario_id}: "
                 f"{SCENARIOS[result.scenario_id].name_vi}",
                 fontsize=13, fontweight="bold", y=1.01)
    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / f"m5_risk_dashboard_{result.scenario_id}.png",
                    dpi=150, bbox_inches="tight")
        logger.info("Đã lưu: outputs/m5_risk_dashboard_%s.png", result.scenario_id)
    return fig


def plot_scenario_risk_comparison(df_cmp: pd.DataFrame, save: bool = True) -> plt.Figure:
    """So sánh Fan Chart GDP 2030 qua 5 kịch bản.

    Args:
        df_cmp: DataFrame từ RiskAnalyzer.compare_scenario_risks().
        save: Lưu PNG.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"S1":"#6B7280","S2":"#3B82F6","S3":"#8B5CF6","S4":"#10B981","S5":"#F59E0B"}

    # ── Trái: P5/P50/P95 per scenario ─────────────────────────
    x = np.arange(len(df_cmp))
    for xi, row in df_cmp.iterrows():
        c = colors.get(row["scenario_id"], "gray")
        ax1.vlines(xi, row["p5_gdp_tn"], row["p95_gdp_tn"],
                   color=c, lw=6, alpha=0.35)
        ax1.plot(xi, row["p50_gdp_tn"], "o", color=c, ms=10, zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_cmp["scenario_name"], rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("GDP 2030 (nghìn tỷ VND)")
    ax1.set_title("Khoảng tin cậy GDP 2030\n(Dải P5–P95, Điểm = P50)",
                  fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # ── Phải: Xác suất hụt mục tiêu ───────────────────────────
    bar_c = [colors.get(s, "gray") for s in df_cmp["scenario_id"]]
    ax2.bar(df_cmp["scenario_name"], df_cmp["prob_miss_pct"],
            color=bar_c, alpha=0.85, edgecolor="white")
    ax2.axhline(20, color="red", ls="--", lw=1.5, label="Ngưỡng rủi ro 20%")
    ax2.set_xticklabels(df_cmp["scenario_name"], rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("P(GDP 2030 < Mục tiêu) %")
    ax2.set_title("Xác suất Hụt Mục tiêu GDP 2030\n(Thấp hơn = An toàn hơn)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("So sánh Rủi ro GDP 2030 theo 5 Kịch bản Chính sách",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / "m5_scenario_risk_comparison.png", dpi=150)
    return fig


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def run_m5_analysis() -> Dict:
    """Chạy toàn bộ phân tích M5."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    from typing import Dict

    print("\n" + "="*60)
    print("  MODULE M5 — Phân tích Rủi ro & Stress Testing")
    print("  Monte Carlo 10,000 mô phỏng | 5 Kịch bản")
    print("="*60)

    analyzer = RiskAnalyzer(n_simulations=10_000, random_seed=42)

    # 1. Phân tích chi tiết kịch bản S5 (tối ưu)
    result_s5 = analyzer.run(scenario_id="S5", include_stress=True)
    analyzer.print_summary(result_s5)

    # 2. So sánh rủi ro 5 kịch bản
    print("\n--- So sánh Rủi ro 5 Kịch bản ---")
    df_risk_cmp = analyzer.compare_scenario_risks()
    print(df_risk_cmp[[
        "scenario_id","scenario_name","p50_gdp_tn",
        "var_95","prob_miss_pct","risk_premium_tn"
    ]].to_string(index=False))

    # 3. Vẽ biểu đồ
    plot_risk_dashboard(result_s5, save=True)
    plot_scenario_risk_comparison(df_risk_cmp, save=True)
    print("\n✅ Biểu đồ M5 đã lưu vào outputs/")

    return {
        "analyzer": analyzer,
        "result_s5": result_s5,
        "risk_comparison": df_risk_cmp,
    }


if __name__ == "__main__":
    run_m5_analysis()
