"""
m6_dashboard.py — Module M6: Nghiệp vụ Dashboard & Trực quan hóa
=================================================================
Module này đóng vai trò tầng nghiệp vụ (business logic layer)
của Dashboard AIDEOM-VN, tách biệt hoàn toàn khỏi framework UI
(Streamlit/Dash). Thiết kế theo nguyên tắc Separation of Concerns:

    M1–M5  →  tính toán mô hình
    M6     →  tổng hợp, định dạng, và chuẩn bị dữ liệu cho UI
    app.py →  render UI (chỉ gọi M6, không gọi M1–M5 trực tiếp)

Lợi ích:
  - Testable độc lập với UI framework
  - Có thể dùng lại cho API REST, CLI, hoặc Jupyter
  - Cache kết quả tính toán nặng tại tầng M6

Chức năng chính:
  1. build_kpi_table()     — Bảng KPI tổng hợp 5 kịch bản
  2. build_gdp_timeseries()— Chuỗi thời gian GDP cho biểu đồ
  3. build_allocation_df() — Ma trận phân bổ ngân sách sẵn UI
  4. build_risk_summary()  — Tóm tắt rủi ro cho dashboard
  5. build_labor_summary() — Tóm tắt thị trường lao động
  6. build_readiness_df()  — Xếp hạng sẵn sàng AI
  7. generate_policy_alerts()— Cảnh báo chính sách tự động
  8. export_excel_report() — Xuất báo cáo Excel đa sheet

Author: AIDEOM-VN Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    SCENARIOS, KPI_TARGETS_2030, REGION_NAMES_VI,
    ITEM_NAMES_VI, OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Dataclass cảnh báo chính sách
# ═══════════════════════════════════════════════════════════════

@dataclass
class PolicyAlert:
    """Cảnh báo chính sách từ kết quả mô hình.

    Attributes:
        level:   "critical" | "warning" | "info" | "success"
        module:  Module nguồn ("M1"…"M5")
        title:   Tiêu đề ngắn
        message: Mô tả chi tiết và khuyến nghị
        value:   Giá trị số liên quan (nếu có)
        target:  Mục tiêu chính sách liên quan
    """
    level: str
    module: str
    title: str
    message: str
    value: Optional[float] = None
    target: Optional[str] = None

    @property
    def emoji(self) -> str:
        return {"critical": "🔴", "warning": "🟡",
                "info": "🔵", "success": "🟢"}.get(self.level, "⚪")


# ═══════════════════════════════════════════════════════════════
# Class chính: DashboardBuilder
# ═══════════════════════════════════════════════════════════════

class DashboardBuilder:
    """Tổng hợp kết quả M1–M5 thành dữ liệu sẵn sàng cho UI.

    Nhận vào PipelineOutputs và xuất ra các DataFrame/Dict đã
    được định dạng, làm sạch và sẵn sàng render.

    Args:
        pipeline_outputs: Kết quả từ AIDEOMPipeline.run_all().

    Example:
        >>> from src.pipeline import AIDEOMPipeline
        >>> pipe = AIDEOMPipeline()
        >>> outputs = pipe.run_all()
        >>> builder = DashboardBuilder(outputs)
        >>> kpi_df = builder.build_kpi_table()
        >>> alerts = builder.generate_policy_alerts()
    """

    SCENARIO_COLORS = {
        "S1": "#6B7280", "S2": "#3B82F6",
        "S3": "#8B5CF6", "S4": "#10B981", "S5": "#F59E0B",
    }

    def __init__(self, pipeline_outputs: Any):
        self.out = pipeline_outputs
        self._validate_outputs()

    def _validate_outputs(self) -> None:
        """Kiểm tra pipeline_outputs có đủ dữ liệu cần thiết."""
        for module in ["m1", "m2", "m3", "m4", "m5"]:
            if not getattr(self.out, module, None):
                logger.warning("Module %s chưa có kết quả", module.upper())

    # ─── 1. Bảng KPI tổng hợp ──────────────────────────────────

    def build_kpi_table(self, year: int = 2030) -> pd.DataFrame:
        """Xây dựng bảng KPI so sánh 5 kịch bản.

        Cột bao gồm: GDP dự báo, tăng trưởng bình quân, kinh tế số,
        AI capacity, nhân lực, Z* LP, NetJob, VaR 95%, P(hụt mục tiêu).

        Args:
            year: Năm mục tiêu so sánh (mặc định 2030).

        Returns:
            DataFrame 5 hàng (kịch bản) × ~12 cột KPI.
        """
        m1 = self.out.m1
        m3 = self.out.m3
        m4 = self.out.m4
        m5 = self.out.m5

        rows = []
        for sid, scenario in SCENARIOS.items():
            fr = m1["forecast_results"].get(sid)
            if fr is None:
                continue

            # Tìm chỉ số năm mục tiêu
            year_idx = (year - 2026) if year >= 2026 else -1
            year_idx = min(year_idx, len(fr.gdp) - 1)

            # M1 metrics
            gdp_target  = fr.gdp[year_idx]
            dig_target  = fr.digital[year_idx]
            ai_target   = fr.ai_cap[year_idx]
            hc_target   = fr.human_cap[year_idx]

            # Tăng trưởng bình quân
            from src.config import INITIAL_CONDITIONS_2026
            n_years = year_idx + 1
            avg_growth = (
                (gdp_target / INITIAL_CONDITIONS_2026["Y0"]) ** (1 / max(n_years, 1)) - 1
            ) * 100 if gdp_target > 0 else 0

            # M3 metrics
            z_star = 0.0
            if m3 and "results_by_scenario" in m3:
                r3 = m3["results_by_scenario"].get(sid)
                if r3:
                    z_star = r3.objective_value

            # M5 metrics
            var_95 = prob_miss = 0.0
            if m5 and "risk_by_scenario" in m5:
                r5 = m5["risk_by_scenario"].get(sid)
                if r5:
                    var_95    = r5.var_95
                    prob_miss = r5.prob_below_target * 100

            rows.append({
                "Kịch bản":            f"{sid}: {scenario.name_vi}",
                "scenario_id":          sid,
                f"GDP {year} (nghìn tỷ)": round(gdp_target, 0),
                "Tăng trưởng (%/năm)": round(avg_growth, 2),
                "Kinh tế số (%)":      round(dig_target, 1),
                "DN Công nghệ (nghìn)": round(ai_target, 1),
                "Nhân lực đào tạo (%)": round(hc_target, 1),
                "Z* LP (tỷ VND)":      round(z_star, 0),
                "VaR 95% (nghìn tỷ)":  round(var_95, 0),
                "P(hụt mục tiêu) %":   round(prob_miss, 1),
                "Đánh giá tổng thể":   self._score_scenario(avg_growth, dig_target, prob_miss),
            })

        df = pd.DataFrame(rows)
        df = df.drop(columns=["scenario_id"])
        return df

    def _score_scenario(
        self, growth: float, digital: float, risk: float
    ) -> str:
        """Đánh giá tổng thể kịch bản bằng emoji."""
        score = 0
        if growth >= 7.0: score += 2
        elif growth >= 5.5: score += 1
        if digital >= 25: score += 2
        elif digital >= 20: score += 1
        if risk <= 10: score += 2
        elif risk <= 25: score += 1
        return {0:"❌ Yếu", 1:"⚠️ Trung bình", 2:"⚠️ Khá",
                3:"✅ Tốt", 4:"✅ Tốt", 5:"🏆 Xuất sắc",
                6:"🏆 Xuất sắc"}.get(score, "N/A")

    # ─── 2. Chuỗi thời gian GDP ────────────────────────────────

    def build_gdp_timeseries(self) -> pd.DataFrame:
        """Chuỗi thời gian GDP 2020–2035 cho tất cả kịch bản.

        Returns:
            DataFrame dạng long: year, scenario_id, scenario_name,
            gdp, growth_rate, digital_pct, ai_cap, human_cap, color.
        """
        from src.data_loader import load_macro

        rows = []

        # Lịch sử 2020–2025
        try:
            hist = load_macro()
            for _, row in hist.iterrows():
                rows.append({
                    "year": int(row["year"]),
                    "scenario_id": "HIST",
                    "scenario_name": "Lịch sử",
                    "gdp": row["GDP_trillion_VND"],
                    "growth_rate": 0.0,
                    "digital_pct": row["D_digital_pct"],
                    "ai_cap": row["AI_tech_firms_thousand"],
                    "human_cap": row["H_trained_labor_pct"],
                    "color": "#000000",
                })
        except Exception as e:
            logger.warning("Không nạp được lịch sử: %s", e)

        # Dự báo 2026–2035
        m1 = self.out.m1
        if not m1:
            return pd.DataFrame(rows)

        for sid, fr in m1["forecast_results"].items():
            for i, yr in enumerate(fr.years):
                rows.append({
                    "year": yr,
                    "scenario_id": sid,
                    "scenario_name": SCENARIOS[sid].name_vi,
                    "gdp": round(fr.gdp[i], 1),
                    "growth_rate": round(fr.growth_rates[i], 2),
                    "digital_pct": round(fr.digital[i], 2),
                    "ai_cap": round(fr.ai_cap[i], 1),
                    "human_cap": round(fr.human_cap[i], 2),
                    "color": self.SCENARIO_COLORS.get(sid, "#374151"),
                })

        return pd.DataFrame(rows)

    # ─── 3. Ma trận phân bổ ngân sách ──────────────────────────

    def build_allocation_df(
        self, scenario_id: str = "S5"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ma trận phân bổ ngân sách tối ưu cho kịch bản đã chọn.

        Args:
            scenario_id: Kịch bản cần truy xuất.

        Returns:
            Tuple: (wide_df, long_df)
              wide_df — ma trận [vùng × hạng mục] với tên tiếng Việt
              long_df — dạng long cho biểu đồ grouped bar
        """
        m3 = self.out.m3
        if not m3 or "results_by_scenario" not in m3:
            return pd.DataFrame(), pd.DataFrame()

        result = m3["results_by_scenario"].get(scenario_id)
        if result is None:
            return pd.DataFrame(), pd.DataFrame()

        mat = result.allocation_matrix.copy()

        # Đổi tên sang tiếng Việt
        mat.index   = [REGION_NAMES_VI.get(r, r) for r in mat.index]
        mat.columns = [ITEM_NAMES_VI.get(j, j) for j in mat.columns]
        mat.index.name = "Vùng KT-XH"
        mat["Tổng"] = mat.sum(axis=1)

        # Dạng long
        long_df = mat.drop(columns=["Tổng"]).reset_index().melt(
            id_vars="Vùng KT-XH",
            var_name="Hạng mục",
            value_name="Ngân sách (tỷ VND)",
        )

        return mat, long_df

    # ─── 4. Tóm tắt rủi ro ─────────────────────────────────────

    def build_risk_summary(
        self, scenario_id: str = "S5"
    ) -> Dict[str, Any]:
        """Tóm tắt rủi ro cho một kịch bản.

        Returns:
            Dict:
              percentiles   — P5/P25/P50/P75/P95
              var_95        — Value at Risk
              cvar_95       — Conditional VaR
              prob_miss     — Xác suất hụt mục tiêu (%)
              sensitivity_df— Phân tích độ nhạy nguồn rủi ro
              stress_df     — Kết quả stress test
              fan_df        — Dữ liệu cho fan chart
        """
        m5 = self.out.m5
        if not m5:
            return {}

        risk_by = m5.get("risk_by_scenario", {})
        result  = risk_by.get(scenario_id) or m5.get("result_s5")
        if result is None:
            return {}

        p = result.percentiles
        gdp = result.gdp_simulations

        # Fan chart data: phân vị tích lũy
        sorted_gdp = np.sort(gdp)
        cdf = np.arange(1, len(sorted_gdp) + 1) / len(sorted_gdp) * 100
        fan_df = pd.DataFrame({"gdp": sorted_gdp, "cdf_pct": cdf})

        # Stress test dataframe
        stress_rows = []
        for k, v in result.stress_results.items():
            from src.m5_risk import RiskAnalyzer
            label = RiskAnalyzer.STRESS_SCENARIOS.get(k, {}).get("label", k)
            chg   = (v / max(p[50], 1) - 1) * 100
            level = ("success" if chg >= -5
                     else "warning" if chg >= -15 else "critical")
            stress_rows.append({
                "Kịch bản": label,
                "GDP 2030 (nghìn tỷ)": round(v, 0),
                "vs P50 (%)": round(chg, 1),
                "Mức rủi ro": {"success":"🟢 An toàn",
                               "warning":"🟡 Chú ý",
                               "critical":"🔴 Rủi ro cao"}[level],
            })

        return {
            "scenario_id":  scenario_id,
            "percentiles":  p,
            "var_95":       result.var_95,
            "cvar_95":      result.cvar_95,
            "prob_miss":    result.prob_below_target * 100,
            "n_sims":       result.n_simulations,
            "sensitivity":  result.sensitivity,
            "stress_df":    pd.DataFrame(stress_rows),
            "fan_df":       fan_df,
            "gdp_sims":     gdp,
        }

    # ─── 5. Tóm tắt lao động ───────────────────────────────────

    def build_labor_summary(self) -> Dict[str, Any]:
        """Tóm tắt phân tích thị trường lao động từ M4.

        Returns:
            Dict:
              sector_df     — DataFrame chi tiết 8 ngành
              total_netjob  — Tổng NetJob (nghìn việc)
              risk_matrix_df— Dữ liệu ma trận rủi ro-cơ hội
              threshold_df  — Ngưỡng đào tạo tối thiểu
        """
        m4 = self.out.m4
        if not m4:
            return {}

        result = m4["result_optimal"]
        sim    = m4["simulator"]

        # DataFrame ngành
        sector_rows = []
        for i, name in enumerate(result.sector_names):
            sector_rows.append({
                "Ngành": name,
                "Rủi ro TĐH (%)": round(sim.risk[i] * 100, 1),
                "Đầu tư AI (tỷ)": round(result.allocation_ai[i], 0),
                "Đầu tư Đào tạo (tỷ)": round(result.allocation_h[i], 0),
                "Việc làm mới AI (k)": round(result.new_jobs[i], 1),
                "Nâng cấp kỹ năng (k)": round(result.upgrade_jobs[i], 1),
                "Bị thay thế (k)": round(result.displaced_jobs[i], 1),
                "NetJob ròng (k)": round(result.net_jobs[i], 1),
                "Đánh giá": ("🟢 Tốt" if result.net_jobs[i] > 0
                             else "🔴 Rủi ro"),
            })
        sector_df = pd.DataFrame(sector_rows)

        # Ngưỡng đào tạo
        threshold_rows = []
        for i, s in enumerate(sim.sectors):
            thr = sim._find_training_threshold(i)
            threshold_rows.append({
                "Ngành": s["name"],
                "Lao động (tr người)": s["labor_M"],
                "Rủi ro (%)": round(s["risk"] * 100, 1),
                "x_H tối thiểu (tỷ VND)": round(thr, 0),
                "Ưu tiên": ("🔴 Cao" if s["risk"] > 0.40
                            else "🟡 Trung bình" if s["risk"] > 0.25
                            else "🟢 Thấp"),
            })

        return {
            "sector_df":    sector_df,
            "total_netjob": result.total_net,
            "threshold_df": pd.DataFrame(threshold_rows),
            "budget_cmp":   m4.get("budget_comparison", pd.DataFrame()),
        }

    # ─── 6. Xếp hạng sẵn sàng AI ──────────────────────────────

    def build_readiness_df(self) -> Dict[str, pd.DataFrame]:
        """Xếp hạng TOPSIS vùng và ngành từ M2.

        Returns:
            Dict: {"region": df_region, "sector": df_sector}
            Mỗi DataFrame có cột: rank, name, score_expert,
            score_entropy, top3_flag, color.
        """
        m2 = self.out.m2
        if not m2:
            return {"region": pd.DataFrame(), "sector": pd.DataFrame()}

        result = {}
        for etype in ["region", "sector"]:
            key = f"{etype}_result"
            r   = m2.get(key)
            if r is None:
                result[etype] = pd.DataFrame()
                continue

            df_exp = r.ranking_expert.copy()
            df_ent = r.ranking_entropy.copy()

            # Merge expert + entropy scores
            merged = df_exp[["name", "rank", "topsis_score"]].rename(
                columns={"topsis_score": "score_expert"}
            )
            ent_scores = df_ent.set_index("name")["topsis_score"].rename("score_entropy")
            merged = merged.join(ent_scores, on="name")
            merged["top3_flag"] = merged["rank"] <= 3
            merged["color"] = merged["rank"].map(
                {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
            ).fillna("#E5E7EB")

            result[etype] = merged

        return result

    # ─── 7. Cảnh báo chính sách tự động ───────────────────────

    def generate_policy_alerts(
        self, scenario_id: str = "S5"
    ) -> List[PolicyAlert]:
        """Phát hiện và tạo cảnh báo chính sách từ kết quả mô hình.

        Logic phát sinh cảnh báo:
          - Tăng trưởng < 7%/năm → warning
          - Kinh tế số < 25% năm 2030 → warning
          - P(hụt mục tiêu) > 20% → critical
          - NetJob âm ngành nào → critical
          - Chi phí equity > 10% Z* → info
          - Kinh tế số vượt 30% → success

        Args:
            scenario_id: Kịch bản cần đánh giá.

        Returns:
            Danh sách PolicyAlert được sắp xếp theo mức độ nghiêm trọng.
        """
        alerts: List[PolicyAlert] = []

        # ── M1: Tăng trưởng GDP ─────────────────────────────
        m1 = self.out.m1
        if m1:
            cmp_df = m1.get("comparison_df")
            if cmp_df is not None:
                row = cmp_df[cmp_df["scenario_id"] == scenario_id]
                if not row.empty:
                    growth = float(row["avg_growth_pct"].iloc[0])
                    if growth < 7.0:
                        alerts.append(PolicyAlert(
                            level="warning", module="M1",
                            title="Tăng trưởng dưới mục tiêu",
                            message=(f"Kịch bản {scenario_id} dự báo tăng trưởng "
                                     f"{growth:.1f}%/năm, thấp hơn mục tiêu 7% "
                                     f"(Văn kiện ĐH XIII). Cần tăng đầu tư AI và số hóa."),
                            value=growth, target="7.0%/năm",
                        ))
                    else:
                        alerts.append(PolicyAlert(
                            level="success", module="M1",
                            title="Tăng trưởng đạt mục tiêu",
                            message=f"GDP tăng trưởng {growth:.1f}%/năm — đạt mục tiêu 7%.",
                            value=growth,
                        ))

            # Kiểm tra mục tiêu kinh tế số 30%
            fr = m1["forecast_results"].get(scenario_id)
            if fr and len(fr.digital) >= 5:
                d_2030 = fr.digital[4]
                if d_2030 < 25:
                    alerts.append(PolicyAlert(
                        level="critical", module="M1",
                        title="Kinh tế số hụt xa mục tiêu 2030",
                        message=(f"Kinh tế số dự báo đạt {d_2030:.1f}% GDP năm 2030, "
                                 f"cách mục tiêu 30% (QĐ 749) còn {30-d_2030:.1f} điểm %. "
                                 f"Cần tăng mạnh đầu tư hạng mục D và H."),
                        value=d_2030, target="30% GDP",
                    ))
                elif d_2030 >= 30:
                    alerts.append(PolicyAlert(
                        level="success", module="M1",
                        title="Mục tiêu kinh tế số 30% đạt được",
                        message=f"Kinh tế số đạt {d_2030:.1f}% GDP — vượt mục tiêu QĐ 749.",
                        value=d_2030,
                    ))
                else:
                    alerts.append(PolicyAlert(
                        level="warning", module="M1",
                        title="Kinh tế số đang tiệm cận mục tiêu",
                        message=(f"Kinh tế số {d_2030:.1f}% — còn {30-d_2030:.1f} điểm % "
                                 f"để đạt mục tiêu 30% (QĐ 749/QĐ-TTg)."),
                        value=d_2030,
                    ))

        # ── M3: Chi phí equity ──────────────────────────────
        m3 = self.out.m3
        if m3:
            eq_cost = m3.get("equity_cost", 0)
            r_opt   = m3.get("results_by_scenario", {}).get(scenario_id)
            if r_opt and eq_cost:
                pct = eq_cost / max(r_opt.objective_value + eq_cost, 1) * 100
                if pct > 10:
                    alerts.append(PolicyAlert(
                        level="warning", module="M3",
                        title=f"Chi phí công bằng vùng cao ({pct:.1f}%)",
                        message=(f"Ràng buộc equity làm giảm GDP gain "
                                 f"{eq_cost:,.0f} tỷ VND ({pct:.1f}% Z*). "
                                 f"Cân nhắc nới λ từ 0.62 xuống 0.55 để giảm chi phí."),
                        value=pct,
                    ))
                else:
                    alerts.append(PolicyAlert(
                        level="info", module="M3",
                        title=f"Chi phí công bằng vùng hợp lý ({pct:.1f}%)",
                        message=(f"Ràng buộc equity chỉ làm giảm {pct:.1f}% GDP gain. "
                                 f"Mức chi phí này chấp nhận được về mặt chính sách."),
                        value=pct,
                    ))

        # ── M4: NetJob ──────────────────────────────────────
        m4 = self.out.m4
        if m4:
            result_labor = m4.get("result_optimal")
            if result_labor:
                neg_sectors = [
                    result_labor.sector_names[i]
                    for i in range(len(result_labor.net_jobs))
                    if result_labor.net_jobs[i] < -0.5
                ]
                if neg_sectors:
                    alerts.append(PolicyAlert(
                        level="critical", module="M4",
                        title=f"NetJob âm tại {len(neg_sectors)} ngành",
                        message=(f"Các ngành có nguy cơ mất việc làm ròng: "
                                 f"{', '.join(neg_sectors[:3])}. "
                                 f"Cần tăng ngân sách đào tạo lại lao động ngay."),
                        value=float(min(result_labor.net_jobs)),
                    ))
                else:
                    alerts.append(PolicyAlert(
                        level="success", module="M4",
                        title=f"Tất cả ngành duy trì NetJob dương",
                        message=(f"Tổng NetJob ròng: {result_labor.total_net:.0f}k việc. "
                                 f"Không ngành nào có NetJob âm."),
                        value=result_labor.total_net,
                    ))

        # ── M5: Rủi ro ──────────────────────────────────────
        m5 = self.out.m5
        if m5:
            risk_by = m5.get("risk_by_scenario", {})
            r5 = risk_by.get(scenario_id) or m5.get("result_s5")
            if r5:
                prob_miss = r5.prob_below_target * 100
                if prob_miss > 20:
                    alerts.append(PolicyAlert(
                        level="critical", module="M5",
                        title=f"Rủi ro hụt mục tiêu cao ({prob_miss:.1f}%)",
                        message=(f"Xác suất GDP 2030 không đạt mục tiêu: {prob_miss:.1f}%. "
                                 f"Cần tăng dự phòng ngân sách và đầu tư vào nhân lực số "
                                 f"(hệ số kháng cú sốc cao nhất theo M5)."),
                        value=prob_miss, target="< 20%",
                    ))
                elif prob_miss > 10:
                    alerts.append(PolicyAlert(
                        level="warning", module="M5",
                        title=f"Rủi ro trung bình ({prob_miss:.1f}%)",
                        message=(f"Xác suất hụt mục tiêu {prob_miss:.1f}% — ở ngưỡng chú ý. "
                                 f"Rủi ro chính: Địa-Chính trị và Gián đoạn Công nghệ."),
                        value=prob_miss,
                    ))
                else:
                    alerts.append(PolicyAlert(
                        level="success", module="M5",
                        title=f"Rủi ro thấp ({prob_miss:.1f}%)",
                        message=f"Xác suất hụt mục tiêu chỉ {prob_miss:.1f}% — rủi ro được kiểm soát tốt.",
                        value=prob_miss,
                    ))

        # Sắp xếp: critical → warning → info → success
        priority = {"critical": 0, "warning": 1, "info": 2, "success": 3}
        alerts.sort(key=lambda a: priority.get(a.level, 9))
        return alerts

    # ─── 8. Xuất báo cáo Excel ─────────────────────────────────

    def export_excel_report(
        self,
        scenario_id: str = "S5",
        output_path: Optional[Path] = None,
    ) -> Path:
        """Xuất báo cáo tổng hợp sang Excel đa sheet.

        Sheet structure:
          1. KPI Summary   — bảng KPI 5 kịch bản
          2. GDP Forecast  — chuỗi thời gian GDP
          3. Allocation    — phân bổ ngân sách tối ưu
          4. Labor         — phân tích lao động 8 ngành
          5. Risk          — tóm tắt rủi ro
          6. Readiness     — xếp hạng TOPSIS vùng & ngành

        Args:
            scenario_id: Kịch bản mặc định cho các sheet chi tiết.
            output_path: Đường dẫn lưu file. Mặc định outputs/aideom_vn_report.xlsx.

        Returns:
            Path của file Excel đã tạo.
        """
        try:
            import openpyxl
        except ImportError:
            raise ImportError("Cài openpyxl: pip install openpyxl")

        path = output_path or OUTPUTS_DIR / "aideom_vn_report.xlsx"
        OUTPUTS_DIR.mkdir(exist_ok=True)

        writer = pd.ExcelWriter(path, engine="openpyxl")

        # Sheet 1: KPI Summary
        kpi_df = self.build_kpi_table()
        kpi_df.to_excel(writer, sheet_name="KPI_Summary", index=False)

        # Sheet 2: GDP Forecast (pivot)
        ts_df = self.build_gdp_timeseries()
        if not ts_df.empty:
            pivot = ts_df.pivot_table(
                index="year", columns="scenario_id", values="gdp", aggfunc="first"
            )
            pivot.to_excel(writer, sheet_name="GDP_Forecast")

        # Sheet 3: Allocation
        wide_df, _ = self.build_allocation_df(scenario_id)
        if not wide_df.empty:
            wide_df.to_excel(writer, sheet_name="Budget_Allocation")

        # Sheet 4: Labor
        labor = self.build_labor_summary()
        if labor.get("sector_df") is not None and not labor["sector_df"].empty:
            labor["sector_df"].to_excel(writer, sheet_name="Labor_Analysis", index=False)
        if labor.get("threshold_df") is not None and not labor["threshold_df"].empty:
            labor["threshold_df"].to_excel(writer, sheet_name="Training_Threshold", index=False)

        # Sheet 5: Risk
        risk = self.build_risk_summary(scenario_id)
        if risk.get("stress_df") is not None and not risk["stress_df"].empty:
            risk["stress_df"].to_excel(writer, sheet_name="Risk_StressTest", index=False)
        if risk.get("sensitivity") is not None:
            risk["sensitivity"].to_excel(writer, sheet_name="Risk_Sensitivity", index=False)

        # Sheet 6: Readiness
        ready = self.build_readiness_df()
        for etype, df in ready.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=f"Readiness_{etype.title()}", index=False)

        # Sheet 7: Policy Alerts
        alerts = self.generate_policy_alerts(scenario_id)
        alert_rows = [{
            "Module": a.module, "Mức": a.level.upper(),
            "Tiêu đề": a.title, "Khuyến nghị": a.message,
            "Giá trị": a.value, "Mục tiêu": a.target,
        } for a in alerts]
        pd.DataFrame(alert_rows).to_excel(
            writer, sheet_name="Policy_Alerts", index=False
        )

        # Metadata
        meta = pd.DataFrame([{
            "Thông tin": "Mô hình AIDEOM-VN",
            "Giá trị": "AI-Driven Economic Optimization Model for Vietnam",
        }, {
            "Thông tin": "Kịch bản mặc định",
            "Giá trị": f"{scenario_id}: {SCENARIOS[scenario_id].name_vi}",
        }, {
            "Thông tin": "Ngày xuất báo cáo",
            "Giá trị": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }, {
            "Thông tin": "Nguồn dữ liệu",
            "Giá trị": "GSO/NSO, World Bank, Bộ KH&CN, Bộ TT&TT 2024-2025",
        }, {
            "Thông tin": "Tài liệu tham chiếu",
            "Giá trị": "Nghị quyết 57-NQ/TW, QĐ 749, QĐ 127, QĐ 411",
        }])
        meta.to_excel(writer, sheet_name="Metadata", index=False)

        writer.close()
        logger.info("Đã xuất báo cáo Excel: %s", path)
        return path


# ═══════════════════════════════════════════════════════════════
# Hàm tiện ích độc lập (không cần pipeline)
# ═══════════════════════════════════════════════════════════════

def format_vnd(value: float, unit: str = "tỷ") -> str:
    """Định dạng số tiền VND cho hiển thị.

    Args:
        value: Giá trị số.
        unit: "tỷ", "nghìn tỷ", "triệu".

    Returns:
        Chuỗi đã định dạng, ví dụ "64,195 tỷ VND".
    """
    return f"{value:,.0f} {unit} VND"


def get_scenario_color(scenario_id: str) -> str:
    """Trả về mã màu HEX cho kịch bản."""
    colors = {
        "S1": "#6B7280", "S2": "#3B82F6",
        "S3": "#8B5CF6", "S4": "#10B981", "S5": "#F59E0B",
    }
    return colors.get(scenario_id, "#374151")


def compute_scenario_radar_data(pipeline_outputs: Any) -> Dict:
    """Dữ liệu radar chart chuẩn hóa cho 5 kịch bản.

    Các chiều: GDP, Số hóa, AI, Nhân lực, Z* LP, An toàn rủi ro.

    Args:
        pipeline_outputs: Kết quả PipelineOutputs.

    Returns:
        Dict {scenario_id: [v1, v2, v3, v4, v5, v6]} giá trị trong [0,1].
    """
    from src.config import INITIAL_CONDITIONS_2026

    m1 = getattr(pipeline_outputs, "m1", {})
    m3 = getattr(pipeline_outputs, "m3", {})
    m5 = getattr(pipeline_outputs, "m5", {})

    radar = {}
    for sid in SCENARIOS:
        fr = m1.get("forecast_results", {}).get(sid)
        if fr is None:
            continue
        ic = INITIAL_CONDITIONS_2026
        n  = min(4, len(fr.gdp) - 1)

        gdp_norm = min(fr.gdp[n] / 20000, 1.0)
        dig_norm = min(fr.digital[n] / 40, 1.0)
        ai_norm  = min(fr.ai_cap[n] / 120, 1.0)
        hc_norm  = min(fr.human_cap[n] / 50, 1.0)

        z_val = 0.0
        if m3 and "results_by_scenario" in m3:
            r3 = m3["results_by_scenario"].get(sid)
            if r3:
                z_val = min(r3.objective_value / 80_000, 1.0)

        safe_val = 1.0
        if m5 and "risk_by_scenario" in m5:
            r5 = m5["risk_by_scenario"].get(sid)
            if r5:
                safe_val = 1.0 - min(r5.prob_below_target * 5, 1.0)

        radar[sid] = [gdp_norm, dig_norm, ai_norm, hc_norm, z_val, safe_val]

    return radar


# ═══════════════════════════════════════════════════════════════
# Entry point — chạy độc lập để kiểm tra
# ═══════════════════════════════════════════════════════════════

def run_m6_analysis() -> None:
    """Chạy M6 độc lập: build pipeline và demo tất cả hàm."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    print("\n" + "="*60)
    print("  MODULE M6 — Dashboard Business Logic")
    print("  Tổng hợp kết quả M1–M5 & Xuất báo cáo")
    print("="*60)

    from src.pipeline import AIDEOMPipeline
    pipe    = AIDEOMPipeline(scenario_id="S5", n_mc_simulations=2_000, verbose=True)
    outputs = pipe.run_all(save_charts=False)

    builder = DashboardBuilder(outputs)

    # 1. KPI Table
    print("\n--- Bảng KPI tổng hợp 2030 ---")
    kpi_df = builder.build_kpi_table()
    print(kpi_df.drop(columns=["Đánh giá tổng thể"]).to_string(index=False))

    # 2. Policy Alerts
    print("\n--- Cảnh báo chính sách (S5) ---")
    for alert in builder.generate_policy_alerts("S5"):
        print(f"  {alert.emoji} [{alert.module}] {alert.title}")
        print(f"     {alert.message[:90]}...")

    # 3. Labor Summary
    labor = builder.build_labor_summary()
    print("\n--- Tóm tắt Lao động ---")
    print(labor["sector_df"][["Ngành","NetJob ròng (k)","Đánh giá"]].to_string(index=False))

    # 4. Export Excel
    print("\n--- Xuất báo cáo Excel ---")
    path = builder.export_excel_report("S5")
    print(f"  ✅ Đã lưu: {path}")

    print("\n✅ Module M6 hoàn thành!")


if __name__ == "__main__":
    run_m6_analysis()
