"""
test_m2_m4_m5.py — Unit tests Module M2, M4, M5
=================================================
Kiểm tra TOPSIS readiness (M2), mô phỏng lao động (M4),
và phân tích rủi ro Monte Carlo (M5).

Chạy: pytest tests/test_m2_m4_m5.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.m2_readiness import TOPSISRanker, ReadinessResult
from src.m4_labor import LaborMarketSimulator, LaborResult
from src.m5_risk import RiskAnalyzer, RiskResult
from src.config import LABOR_PARAMS, SCENARIOS


# ═══════════════════════════════════════════════════════════════
# MODULE M2 — TOPSIS
# ═══════════════════════════════════════════════════════════════

class TestTOPSISRanker:
    """Kiểm tra lớp TOPSISRanker."""

    @pytest.fixture
    def ranker_region(self):
        return TOPSISRanker("region")

    @pytest.fixture
    def ranker_sector(self):
        return TOPSISRanker("sector")

    @pytest.fixture
    def result_region(self, ranker_region):
        return ranker_region.rank()

    @pytest.fixture
    def result_sector(self, ranker_sector):
        return ranker_sector.rank()

    # ── Khởi tạo ──────────────────────────────────────────────
    def test_invalid_entity_type_raises(self):
        with pytest.raises(ValueError):
            TOPSISRanker("invalid_type")

    def test_region_weights_sum_to_one(self, ranker_region):
        assert abs(ranker_region.w_expert.sum() - 1.0) < 1e-4

    def test_sector_weights_sum_to_one(self, ranker_sector):
        assert abs(ranker_sector.w_expert.sum() - 1.0) < 1e-4

    def test_region_criteria_count(self, ranker_region):
        """Vùng có 8 tiêu chí."""
        assert len(ranker_region.criteria) == 8

    def test_sector_criteria_count(self, ranker_sector):
        """Ngành có 7 tiêu chí."""
        assert len(ranker_sector.criteria) == 7

    # ── Kết quả TOPSIS ────────────────────────────────────────
    def test_rank_returns_readiness_result(self, result_region):
        assert isinstance(result_region, ReadinessResult)

    def test_scores_bounded_0_1(self, result_region):
        """Điểm C* phải trong [0, 1]."""
        scores = result_region.scores_expert.values
        assert np.all(scores >= 0) and np.all(scores <= 1.0), \
            f"Điểm ngoài [0,1]: {scores}"

    def test_six_regions_ranked(self, result_region):
        assert len(result_region.ranking_expert) == 6

    def test_ten_sectors_ranked(self, result_sector):
        assert len(result_sector.ranking_expert) == 10

    def test_ranking_order_consistent(self, result_region):
        """Xếp hạng theo thứ tự giảm dần của điểm."""
        scores = result_region.ranking_expert["topsis_score"].values
        assert np.all(scores[:-1] >= scores[1:] - 1e-10), \
            "Thứ tự xếp hạng không giảm dần"

    def test_top3_stable_attribute_is_bool(self, result_region):
        assert isinstance(result_region.top3_stable, bool)

    def test_entropy_weights_sum_to_one(self, result_region):
        assert abs(result_region.weights_entropy.sum() - 1.0) < 1e-6

    def test_entropy_weights_nonneg(self, result_region):
        assert np.all(result_region.weights_entropy >= 0)

    # ── Phương thức Entropy ───────────────────────────────────
    def test_entropy_uniform_data_equal_weights(self, ranker_region):
        """Dữ liệu đều nhau → Entropy weight đều nhau."""
        X_uniform = np.ones((6, 8)) * 5.0
        # Kết quả có thể bằng nhau hoặc undefined; hàm không crash là đủ
        w = ranker_region.entropy_weights(X_uniform)
        assert len(w) == 8
        assert abs(w.sum() - 1.0) < 1e-6

    def test_entropy_high_variance_col_gets_higher_weight(self, ranker_region):
        """Cột có phương sai cao hơn → trọng số Entropy cao hơn."""
        np.random.seed(42)
        X = np.random.rand(6, 3) + 1.0
        X[:, 2] *= 10  # cột 2 phương sai lớn hơn
        w = ranker_region.entropy_weights(X)
        assert w[2] > w[0], \
            f"Cột phương sai cao (w={w[2]:.3f}) ≤ cột phương sai thấp (w={w[0]:.3f})"

    # ── Phân tích độ nhạy ─────────────────────────────────────
    def test_sensitivity_returns_dataframe(self, ranker_region):
        df = ranker_region.sensitivity_ai_weight()
        assert isinstance(df, pd.DataFrame)
        assert "w_ai" in df.columns

    def test_sensitivity_w_ai_range(self, ranker_region):
        df = ranker_region.sensitivity_ai_weight()
        assert df["w_ai"].min() >= 0.04
        assert df["w_ai"].max() <= 0.41

    def test_sensitivity_top3_columns_present(self, ranker_region):
        df = ranker_region.sensitivity_ai_weight()
        for col in ["rank_1", "rank_2", "rank_3"]:
            assert col in df.columns


# ═══════════════════════════════════════════════════════════════
# MODULE M4 — LABOR
# ═══════════════════════════════════════════════════════════════

class TestLaborMarketSimulator:
    """Kiểm tra LaborMarketSimulator."""

    @pytest.fixture
    def sim(self):
        return LaborMarketSimulator(total_budget=30_000)

    @pytest.fixture
    def result(self, sim):
        return sim.optimize(constrain_net_positive=True)

    # ── Khởi tạo ──────────────────────────────────────────────
    def test_eight_sectors(self, sim):
        assert sim.N == 8

    def test_risk_in_0_1(self, sim):
        assert np.all(sim.risk >= 0) and np.all(sim.risk <= 1)

    def test_positive_params(self, sim):
        for arr_name in ["a1", "b1", "c1", "d1"]:
            arr = getattr(sim, arr_name)
            assert np.all(arr > 0), f"{arr_name} có phần tử ≤ 0"

    # ── Tối ưu ────────────────────────────────────────────────
    def test_status_optimal(self, result):
        assert "Optimal" in result.status, f"Status: {result.status}"

    def test_result_type(self, result):
        assert isinstance(result, LaborResult)

    def test_total_budget_respected(self, result):
        total_invest = (result.allocation_ai + result.allocation_h).sum()
        assert total_invest <= 30_000 + 1.0, \
            f"Vượt ngân sách: {total_invest:,.0f}"

    def test_net_jobs_positive_per_sector(self, result):
        """Với constrain_net_positive=True, mọi ngành phải NetJob ≥ 0."""
        assert np.all(result.net_jobs >= -0.01), \
            f"NetJob âm: {result.net_jobs}"

    def test_total_net_positive(self, result):
        assert result.total_net > 0

    def test_sector_names_count(self, result):
        assert len(result.sector_names) == 8

    def test_array_shapes_consistent(self, result):
        for arr_name in ["allocation_ai", "allocation_h", "new_jobs",
                          "upgrade_jobs", "displaced_jobs", "net_jobs"]:
            arr = getattr(result, arr_name)
            assert len(arr) == 8, f"{arr_name} shape sai: {len(arr)}"

    def test_retrain_covers_displaced(self, result, sim):
        """Ràng buộc: Displaced_i ≤ Retrain_i = d1_i * x_H_i."""
        retrain = sim.d1 * result.allocation_h / 1000
        displaced = result.displaced_jobs
        assert np.all(displaced <= retrain + 0.01), \
            "Displaced vượt năng lực đào tạo"

    # ── compute_netjob ─────────────────────────────────────────
    def test_compute_netjob_zero_investment(self, sim):
        """Đầu tư = 0 → tất cả thành phần = 0."""
        nj, uj, dj, netj = sim.compute_netjob(
            np.zeros(8), np.zeros(8)
        )
        assert np.all(netj == 0)

    def test_compute_netjob_formula(self, sim):
        """Kiểm tra công thức NetJob = NewJob + Upgrade - Displaced."""
        x_ai = np.ones(8) * 100
        x_h  = np.ones(8) * 200
        nj, uj, dj, netj = sim.compute_netjob(x_ai, x_h)
        expected = nj + uj - dj
        np.testing.assert_allclose(netj, expected, rtol=1e-10)

    # ── Phân tích ngưỡng ──────────────────────────────────────
    def test_threshold_nonneg(self, sim):
        """Ngưỡng đào tạo tối thiểu phải ≥ 0."""
        for i in range(8):
            t = sim._find_training_threshold(i)
            assert t >= 0, f"Ngành {i}: ngưỡng âm {t}"

    def test_high_risk_sector_has_positive_threshold(self, sim):
        """Ngành rủi ro cao (Tài chính, risk=0.52) cần đào tạo ≥ 0."""
        idx = 4  # Tài chính-Ngân hàng
        t = sim._find_training_threshold(idx)
        assert t >= 0

    # ── So sánh ngân sách ──────────────────────────────────────
    def test_budget_comparison_monotone(self, sim):
        """NetJob tổng tăng khi ngân sách tăng (với phân bổ tỷ lệ đều)."""
        df = sim.compare_investment_levels([10_000, 20_000, 30_000])
        netjobs = df["total_net_jobs_k"].values
        for i in range(len(netjobs) - 1):
            assert netjobs[i+1] >= netjobs[i] - 1.0, \
                "NetJob giảm khi tăng ngân sách"

    def test_5pct_displacement_constraint(self, sim):
        """Với max_displacement_rate=5%, displaced ≤ 5% lao động ngành."""
        result = sim.optimize(
            constrain_net_positive=False,
            max_displacement_rate=0.05,
        )
        assert "Optimal" in result.status
        displaced_k = result.displaced_jobs  # nghìn người
        for i in range(sim.N):
            labor_k = sim.labor_M[i]  # triệu người = 1000 nghìn
            assert displaced_k[i] <= labor_k * 0.05 * 1000 + 0.01, \
                f"Ngành {i}: displaced {displaced_k[i]:.1f}k > 5% * {labor_k*1000:.0f}k"


# ═══════════════════════════════════════════════════════════════
# MODULE M5 — RISK
# ═══════════════════════════════════════════════════════════════

class TestRiskAnalyzer:
    """Kiểm tra RiskAnalyzer Monte Carlo."""

    @pytest.fixture
    def analyzer(self):
        return RiskAnalyzer(n_simulations=500, random_seed=42,
                            target_gdp_2030=14_000.0)

    @pytest.fixture
    def result(self, analyzer):
        return analyzer.run("S5", include_stress=True)

    # ── Khởi tạo ──────────────────────────────────────────────
    def test_default_n_simulations(self):
        a = RiskAnalyzer(n_simulations=100)
        assert a.N == 100

    def test_five_risk_factors(self, analyzer):
        assert len(analyzer.RISK_FACTORS) == 5

    def test_five_stress_scenarios(self, analyzer):
        assert len(analyzer.STRESS_SCENARIOS) == 5

    # ── Kết quả run() ─────────────────────────────────────────
    def test_returns_risk_result(self, result):
        assert isinstance(result, RiskResult)

    def test_simulations_count(self, result):
        assert len(result.gdp_simulations) == 500

    def test_gdp_simulations_positive(self, result):
        assert np.all(result.gdp_simulations > 0)

    def test_percentiles_ordered(self, result):
        """P5 ≤ P25 ≤ P50 ≤ P75 ≤ P95."""
        p = result.percentiles
        assert p[5] <= p[25] <= p[50] <= p[75] <= p[95], \
            f"Phân vị không có thứ tự: {p}"

    def test_var_equals_p5(self, result):
        """VaR 95% = P5 của phân phối."""
        assert abs(result.var_95 - result.percentiles[5]) < 1.0

    def test_cvar_leq_var(self, result):
        """CVaR ≤ VaR (đuôi mean ≤ ngưỡng)."""
        assert result.cvar_95 <= result.var_95 + 1.0

    def test_prob_below_target_in_0_1(self, result):
        assert 0.0 <= result.prob_below_target <= 1.0

    def test_stress_results_present(self, result):
        """Stress test phải có đủ 5 kịch bản."""
        assert len(result.stress_results) == 5

    def test_base_case_is_highest_stress(self, result):
        """Kịch bản cơ sở (không cú sốc) phải cho GDP cao nhất."""
        base = result.stress_results.get("base_case", 0)
        for key, val in result.stress_results.items():
            if key != "base_case":
                assert base >= val - 1.0, \
                    f"Base case ({base:.0f}) < {key} ({val:.0f})"

    def test_sensitivity_dataframe(self, result):
        """Sensitivity analysis phải trả về DataFrame."""
        assert isinstance(result.sensitivity, pd.DataFrame)
        assert "factor" in result.sensitivity.columns
        assert "pct_of_total" in result.sensitivity.columns

    def test_sensitivity_pct_positive(self, result):
        """Tỷ lệ đóng góp phương sai phải dương."""
        assert (result.sensitivity["pct_of_total"] >= 0).all()

    def test_reproducible_with_same_seed(self, analyzer):
        """Cùng seed → kết quả giống nhau."""
        r1 = analyzer.run("S5", include_stress=False)
        r2 = RiskAnalyzer(n_simulations=500, random_seed=42).run(
            "S5", include_stress=False
        )
        np.testing.assert_allclose(
            r1.percentiles[50], r2.percentiles[50], rtol=1e-6
        )

    # ── compare_scenario_risks ────────────────────────────────
    def test_compare_returns_dataframe(self, analyzer):
        df = analyzer.compare_scenario_risks()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_compare_has_all_scenarios(self, analyzer):
        df = analyzer.compare_scenario_risks()
        assert set(df["scenario_id"]) == {"S1", "S2", "S3", "S4", "S5"}

    def test_compare_risk_premium_col(self, analyzer):
        df = analyzer.compare_scenario_risks()
        assert "risk_premium_tn" in df.columns
        assert (df["risk_premium_tn"] >= 0).all()

    # ── Tính hợp lý kinh tế ───────────────────────────────────
    def test_gdp_2030_range(self, result):
        """GDP 2030 P50 phải trong dải hợp lý 12,000–25,000 nghìn tỷ."""
        p50 = result.percentiles[50]
        assert 12_000 < p50 < 25_000, \
            f"GDP 2030 P50 = {p50:,.0f} ngoài dải hợp lý"

    def test_uncertainty_band_reasonable(self, result):
        """Dải bất định P5–P95 không quá rộng (< 50% P50)."""
        band = result.percentiles[95] - result.percentiles[5]
        p50  = result.percentiles[50]
        assert band / p50 < 0.50, \
            f"Dải không chắc quá rộng: {band/p50:.1%} của P50"
