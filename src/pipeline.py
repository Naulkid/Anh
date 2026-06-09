"""
pipeline.py — Orchestrator Pipeline AIDEOM-VN
===============================================
Kết nối và chạy tuần tự M1→M2→M3→M4→M5, chuyển
đầu ra của module trước thành đầu vào của module sau.

Luồng dữ liệu:
  M1 (Forecast) ──▶ forecast_results, comparison_df
        │
        ▼
  M2 (Readiness) ─▶ topsis_regions, topsis_sectors
        │
        ▼
  M3 (Optimize) ──▶ optimal_allocation, scenario_results
        │
        ▼
  M4 (Labor) ─────▶ netjob_result, budget_comparison
        │
        ▼
  M5 (Risk) ──────▶ risk_results, scenario_risks

Usage:
    from src.pipeline import AIDEOMPipeline
    pipe = AIDEOMPipeline()
    outputs = pipe.run_all()
    # hoặc chạy từng bước:
    outputs = pipe.run_step("M1")

Author: AIDEOM-VN Team
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutputs:
    """Tổng hợp kết quả tất cả module.

    Attributes:
        m1: Dict kết quả M1 (model, forecast_results, comparison_df, ...).
        m2: Dict kết quả M2 (region_result, sector_result).
        m3: Dict kết quả M3 (result_optimal, results_by_scenario, ...).
        m4: Dict kết quả M4 (result_optimal, result_constrained, ...).
        m5: Dict kết quả M5 (result_s5, risk_comparison).
        run_time: Thời gian chạy mỗi module (giây).
        errors: Lỗi nếu có trong module nào đó.
    """
    m1: Dict[str, Any] = field(default_factory=dict)
    m2: Dict[str, Any] = field(default_factory=dict)
    m3: Dict[str, Any] = field(default_factory=dict)
    m4: Dict[str, Any] = field(default_factory=dict)
    m5: Dict[str, Any] = field(default_factory=dict)
    run_time: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


class AIDEOMPipeline:
    """Pipeline AIDEOM-VN — chạy tuần tự 5 module phân tích.

    Args:
        scenario_id: Kịch bản mặc định cho tối ưu và rủi ro.
        n_mc_simulations: Số mô phỏng Monte Carlo (M5).
        verbose: In tiến trình chi tiết.

    Example:
        >>> pipe = AIDEOMPipeline(scenario_id="S5")
        >>> outputs = pipe.run_all()
        >>> print(outputs.m1["comparison_df"])
    """

    MODULES = ["M1", "M2", "M3", "M4", "M5"]

    def __init__(
        self,
        scenario_id: str = "S5",
        n_mc_simulations: int = 5_000,
        verbose: bool = True,
    ):
        self.scenario_id = scenario_id
        self.n_mc = n_mc_simulations
        self.verbose = verbose
        self.outputs = PipelineOutputs()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ─── Chạy từng module ──────────────────────────────────────

    def run_m1(self) -> Dict:
        """Chạy Module M1: Dự báo Cobb-Douglas."""
        self._log("\n[M1] Dự báo GDP Cobb-Douglas...")
        t0 = time.time()
        from src.m1_forecasting import CobbDouglasModel, SCENARIOS, INITIAL_CONDITIONS_2026
        import numpy as np

        model = CobbDouglasModel()
        model.fit()
        decomp_df    = model.growth_accounting()
        comparison_df = model.compare_scenarios(year_end=2030)
        forecast_results = {
            sid: model.forecast(2026, 2035, sid) for sid in SCENARIOS
        }

        out = {
            "model":            model,
            "decomp_df":        decomp_df,
            "comparison_df":    comparison_df,
            "forecast_results": forecast_results,
            "tfp_hist":         model.tfp_hist,
            "mape_fit":         model.mape_fit,
        }
        self.outputs.m1 = out
        self.outputs.run_time["M1"] = round(time.time() - t0, 2)
        self._log(f"  ✓ M1 hoàn thành ({self.outputs.run_time['M1']}s) | MAPE={model.mape_fit:.2f}%")
        return out

    def run_m2(self) -> Dict:
        """Chạy Module M2: TOPSIS Readiness."""
        self._log("\n[M2] Xếp hạng sẵn sàng AI/Số (TOPSIS)...")
        t0 = time.time()
        from src.m2_readiness import TOPSISRanker

        region_result = TOPSISRanker("region").rank()
        sector_result = TOPSISRanker("sector").rank()

        out = {
            "region_result":  region_result,
            "sector_result":  sector_result,
            "top3_regions":   region_result.ranking_expert.head(3)["name"].tolist(),
            "top3_sectors":   sector_result.ranking_expert.head(3)["name"].tolist(),
        }
        self.outputs.m2 = out
        self.outputs.run_time["M2"] = round(time.time() - t0, 2)
        self._log(f"  ✓ M2 hoàn thành ({self.outputs.run_time['M2']}s)")
        self._log(f"    Top-3 vùng  : {out['top3_regions']}")
        self._log(f"    Top-3 ngành : {out['top3_sectors']}")
        return out

    def run_m3(self) -> Dict:
        """Chạy Module M3: Tối ưu phân bổ vốn."""
        self._log("\n[M3] Tối ưu phân bổ ngân sách LP...")
        t0 = time.time()
        from src.m3_optimization import BudgetOptimizer

        opt = BudgetOptimizer()
        result_optimal = opt.solve_pulp(with_equity=True, scenario_id=self.scenario_id)
        results_by_scenario = {
            sid: opt.solve_for_scenario(sid, with_equity=True)
            for sid in ["S1", "S2", "S3", "S4", "S5"]
        }
        r_with, r_without, eq_cost = opt.compute_equity_cost()
        sensitivity_df = opt.sensitivity_analysis_budget()

        out = {
            "optimizer":            opt,
            "result_optimal":       result_optimal,
            "results_by_scenario":  results_by_scenario,
            "equity_cost":          eq_cost,
            "sensitivity_df":       sensitivity_df,
        }
        self.outputs.m3 = out
        self.outputs.run_time["M3"] = round(time.time() - t0, 2)
        self._log(f"  ✓ M3 hoàn thành ({self.outputs.run_time['M3']}s) | "
                  f"Z*={result_optimal.objective_value:,.0f} tỷ VND")
        return out

    def run_m4(self) -> Dict:
        """Chạy Module M4: Mô phỏng lao động."""
        self._log("\n[M4] Mô phỏng tác động AI lên lao động...")
        t0 = time.time()
        from src.m4_labor import LaborMarketSimulator

        sim = LaborMarketSimulator(total_budget=30_000)
        result_optimal     = sim.optimize(constrain_net_positive=True)
        result_constrained = sim.optimize(constrain_net_positive=True,
                                          max_displacement_rate=0.05)
        budget_cmp = sim.compare_investment_levels()

        out = {
            "simulator":          sim,
            "result_optimal":     result_optimal,
            "result_constrained": result_constrained,
            "budget_comparison":  budget_cmp,
            "total_netjob":       result_optimal.total_net,
        }
        self.outputs.m4 = out
        self.outputs.run_time["M4"] = round(time.time() - t0, 2)
        self._log(f"  ✓ M4 hoàn thành ({self.outputs.run_time['M4']}s) | "
                  f"NetJob={result_optimal.total_net:.0f}k việc")
        return out

    def run_m5(self) -> Dict:
        """Chạy Module M5: Phân tích rủi ro Monte Carlo."""
        self._log(f"\n[M5] Phân tích rủi ro Monte Carlo ({self.n_mc:,} mô phỏng)...")
        t0 = time.time()
        from src.m5_risk import RiskAnalyzer

        analyzer = RiskAnalyzer(n_simulations=self.n_mc, random_seed=42)
        result_s5    = analyzer.run(scenario_id="S5", include_stress=True)
        risk_cmp_df  = analyzer.compare_scenario_risks()
        # Kết quả chi tiết mỗi kịch bản
        risk_by_scenario = {}
        for sid in ["S1", "S2", "S3", "S4", "S5"]:
            risk_by_scenario[sid] = analyzer.run(sid, include_stress=False)

        out = {
            "analyzer":          analyzer,
            "result_s5":         result_s5,
            "risk_comparison":   risk_cmp_df,
            "risk_by_scenario":  risk_by_scenario,
        }
        self.outputs.m5 = out
        self.outputs.run_time["M5"] = round(time.time() - t0, 2)
        self._log(f"  ✓ M5 hoàn thành ({self.outputs.run_time['M5']}s) | "
                  f"P50_GDP2030={result_s5.percentiles[50]:,.0f}")
        return out

    # ─── Chạy toàn bộ pipeline ─────────────────────────────────

    def run_all(
        self,
        modules: Optional[List[str]] = None,
        save_charts: bool = True,
    ) -> PipelineOutputs:
        """Chạy pipeline đầy đủ M1 → M5.

        Args:
            modules: Danh sách module cần chạy. None = tất cả.
            save_charts: Lưu biểu đồ PNG.

        Returns:
            PipelineOutputs với kết quả đầy đủ.
        """
        to_run = modules or self.MODULES
        t_total = time.time()

        self._log("\n" + "="*62)
        self._log("  AIDEOM-VN PIPELINE — Bắt đầu chạy")
        self._log(f"  Modules: {to_run} | Kịch bản: {self.scenario_id}")
        self._log("="*62)

        runners = {
            "M1": self.run_m1,
            "M2": self.run_m2,
            "M3": self.run_m3,
            "M4": self.run_m4,
            "M5": self.run_m5,
        }

        for module_name in to_run:
            try:
                runners[module_name]()
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                self.outputs.errors[module_name] = error_msg
                logger.error("[%s] LỖI: %s", module_name, error_msg)
                self._log(f"  ✗ {module_name} LỖI: {error_msg}")

        # Tóm tắt
        total_time = round(time.time() - t_total, 1)
        self._log("\n" + "="*62)
        self._log("  PIPELINE HOÀN THÀNH")
        self._log(f"  Tổng thời gian: {total_time}s")
        self._log("  Thời gian mỗi module:")
        for m in to_run:
            t = self.outputs.run_time.get(m, "N/A")
            e = " [LỖI]" if m in self.outputs.errors else ""
            self._log(f"    {m}: {t}s{e}")
        self._log("="*62)

        if save_charts:
            self._generate_all_charts()

        return self.outputs

    def run_step(self, module_name: str) -> Dict:
        """Chạy một module đơn lẻ.

        Args:
            module_name: "M1", "M2", "M3", "M4", hoặc "M5".

        Returns:
            Dict kết quả của module đó.
        """
        runners = {
            "M1": self.run_m1, "M2": self.run_m2,
            "M3": self.run_m3, "M4": self.run_m4,
            "M5": self.run_m5,
        }
        if module_name not in runners:
            raise ValueError(f"Module không hợp lệ: {module_name}. Chọn {list(runners.keys())}")
        return runners[module_name]()

    # ─── Sinh tất cả biểu đồ ───────────────────────────────────

    def _generate_all_charts(self) -> None:
        """Sinh và lưu toàn bộ biểu đồ từ kết quả pipeline."""
        from src.config import OUTPUTS_DIR
        OUTPUTS_DIR.mkdir(exist_ok=True)
        self._log("\n[Charts] Đang sinh biểu đồ...")

        try:
            if self.outputs.m1:
                from src.m1_forecasting import (
                    plot_tfp_trend, plot_gdp_forecast, plot_growth_decomposition
                )
                plot_tfp_trend(self.outputs.m1["model"], save=True)
                plot_gdp_forecast(self.outputs.m1["forecast_results"], save=True)
                plot_growth_decomposition(self.outputs.m1["decomp_df"], save=True)
        except Exception as e:
            logger.warning("M1 charts: %s", e)

        try:
            if self.outputs.m2:
                from src.m2_readiness import (
                    TOPSISRanker, plot_topsis_comparison,
                    plot_entropy_weights, plot_sensitivity_heatmap
                )
                for etype in ["region", "sector"]:
                    r = self.outputs.m2[f"{etype}_result"]
                    plot_topsis_comparison(r, save=True)
                    plot_entropy_weights(r, save=True)
                    ranker = TOPSISRanker(etype)
                    df_s = ranker.sensitivity_ai_weight()
                    if not df_s.empty:
                        plot_sensitivity_heatmap(df_s, r, save=True)
        except Exception as e:
            logger.warning("M2 charts: %s", e)

        try:
            if self.outputs.m3:
                from src.m3_optimization import (
                    plot_allocation_heatmap,
                    plot_scenario_comparison_m3,
                    plot_sensitivity_budget,
                )
                plot_allocation_heatmap(self.outputs.m3["result_optimal"], save=True)
                plot_scenario_comparison_m3(self.outputs.m3["results_by_scenario"], save=True)
                plot_sensitivity_budget(self.outputs.m3["sensitivity_df"], save=True)
        except Exception as e:
            logger.warning("M3 charts: %s", e)

        try:
            if self.outputs.m4:
                from src.m4_labor import plot_netjob_stacked, plot_labor_risk_matrix
                plot_netjob_stacked(self.outputs.m4["result_optimal"], save=True)
                plot_labor_risk_matrix(self.outputs.m4["result_optimal"], save=True)
        except Exception as e:
            logger.warning("M4 charts: %s", e)

        try:
            if self.outputs.m5:
                from src.m5_risk import plot_risk_dashboard, plot_scenario_risk_comparison
                plot_risk_dashboard(self.outputs.m5["result_s5"], save=True)
                plot_scenario_risk_comparison(self.outputs.m5["risk_comparison"], save=True)
        except Exception as e:
            logger.warning("M5 charts: %s", e)

        self._log("  ✓ Biểu đồ đã lưu vào outputs/")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    pipeline = AIDEOMPipeline(scenario_id="S5", n_mc_simulations=2_000)
    outputs  = pipeline.run_all(save_charts=True)
    print(f"\nSố lỗi: {len(outputs.errors)}")
    if outputs.errors:
        for m, e in outputs.errors.items():
            print(f"  {m}: {e}")
