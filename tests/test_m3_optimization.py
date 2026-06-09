"""
test_m3_optimization.py — Unit tests Module M3
===============================================
Kiểm tra tính đúng đắn của LP phân bổ ngân sách,
ràng buộc công bằng vùng, phân tích độ nhạy và
tích hợp với các kịch bản chính sách.

Chạy: pytest tests/test_m3_optimization.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.m3_optimization import BudgetOptimizer, OptimizationResult
from src.config import REGIONS, INVEST_ITEMS, BUDGET_CONSTRAINTS, SCENARIOS


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def optimizer():
    """BudgetOptimizer mặc định."""
    return BudgetOptimizer()


@pytest.fixture
def result_with_equity(optimizer):
    """Nghiệm tối ưu CÓ ràng buộc equity."""
    return optimizer.solve_pulp(with_equity=True, scenario_id="S5")


@pytest.fixture
def result_no_equity(optimizer):
    """Nghiệm tối ưu KHÔNG có ràng buộc equity."""
    return optimizer.solve_pulp(with_equity=False, scenario_id="S5")


# ─────────────────────────────────────────────────────────────
# Test 1: Khởi tạo và cấu hình
# ─────────────────────────────────────────────────────────────

class TestOptimizerInit:
    """Kiểm tra khởi tạo BudgetOptimizer."""

    def test_default_regions(self, optimizer):
        """Phải có đúng 6 vùng kinh tế-xã hội."""
        assert len(optimizer.regions) == 6
        assert "RRD" in optimizer.regions   # Đồng bằng sông Hồng
        assert "SE"  in optimizer.regions   # Đông Nam Bộ

    def test_default_items(self, optimizer):
        """Phải có đúng 4 hạng mục đầu tư."""
        assert len(optimizer.items) == 4
        assert all(j in optimizer.items for j in ["I", "D", "AI", "H"])

    def test_beta_matrix_coverage(self, optimizer):
        """Beta matrix phải có đủ 24 phần tử (6 vùng × 4 hạng mục)."""
        assert len(optimizer.beta) == len(REGIONS) * len(INVEST_ITEMS)

    def test_beta_all_positive(self, optimizer):
        """Tất cả hệ số β phải dương."""
        for key, val in optimizer.beta.items():
            assert val > 0, f"β{key} = {val} phải > 0"

    def test_digital_index_six_regions(self, optimizer):
        """Chỉ số số hóa ban đầu phải có đủ 6 vùng."""
        assert len(optimizer.D0) == 6

    def test_custom_budget_accepted(self):
        """Cho phép truyền budget_constraints tùy chỉnh."""
        custom_bc = BUDGET_CONSTRAINTS.copy()
        custom_bc["total_budget_trillion"] = 60_000
        opt = BudgetOptimizer(budget_constraints=custom_bc)
        assert opt.bc["total_budget_trillion"] == 60_000


# ─────────────────────────────────────────────────────────────
# Test 2: Tính khả thi và tối ưu
# ─────────────────────────────────────────────────────────────

class TestSolveFeasibility:
    """Kiểm tra nghiệm tối ưu hợp lệ."""

    def test_status_optimal(self, result_with_equity):
        """Bài toán phải có nghiệm tối ưu."""
        assert "Optimal" in result_with_equity.status, \
            f"Status không tối ưu: {result_with_equity.status}"

    def test_objective_positive(self, result_with_equity):
        """Giá trị hàm mục tiêu Z* phải dương."""
        assert result_with_equity.objective_value > 0

    def test_objective_reasonable_magnitude(self, result_with_equity):
        """Z* nên nằm trong khoảng hợp lý (30k–100k tỷ VND)."""
        z = result_with_equity.objective_value
        assert 30_000 < z < 100_000, f"Z* = {z} ngoài khoảng hợp lý"

    def test_all_variables_nonneg(self, result_with_equity):
        """Tất cả biến quyết định x_{j,r} phải ≥ 0."""
        mat = result_with_equity.allocation_matrix
        assert (mat.values >= -1e-6).all(), "Có biến âm trong nghiệm"

    def test_status_no_equity(self, result_no_equity):
        """Bài toán không có equity cũng phải khả thi."""
        assert "Optimal" in result_no_equity.status


# ─────────────────────────────────────────────────────────────
# Test 3: Ràng buộc C1–C4 (ngân sách & sàn hạng mục)
# ─────────────────────────────────────────────────────────────

class TestBudgetConstraints:
    """Kiểm tra nghiệm thỏa mãn tất cả ràng buộc ngân sách."""

    def test_c1_total_budget(self, result_with_equity):
        """C1: Tổng đầu tư ≤ ngân sách tổng."""
        total = result_with_equity.allocation_by_region.sum()
        budget = BUDGET_CONSTRAINTS["total_budget_trillion"]
        assert total <= budget + 1.0, \
            f"Vượt ngân sách: {total:,.0f} > {budget:,.0f}"

    def test_c2_floor_per_region(self, result_with_equity):
        """C2: Mỗi vùng nhận ≥ sàn tối thiểu."""
        floor = BUDGET_CONSTRAINTS["min_per_region"]
        for r, total_r in result_with_equity.allocation_by_region.items():
            assert total_r >= floor - 1.0, \
                f"Vùng {r}: {total_r:,.0f} < sàn {floor:,.0f}"

    def test_c3_ceil_per_region(self, result_with_equity):
        """C3: Mỗi vùng nhận ≤ trần tối đa."""
        ceil = BUDGET_CONSTRAINTS["max_per_region"]
        for r, total_r in result_with_equity.allocation_by_region.items():
            assert total_r <= ceil + 1.0, \
                f"Vùng {r}: {total_r:,.0f} > trần {ceil:,.0f}"

    def test_c4_human_capital_floor(self, result_with_equity):
        """C4: Tổng nhân lực số H ≥ 24% ngân sách tổng."""
        mat = result_with_equity.allocation_matrix
        total_H = mat["H"].sum()
        budget = BUDGET_CONSTRAINTS["total_budget_trillion"]
        min_H  = BUDGET_CONSTRAINTS["min_human_capital_pct"] * budget
        assert total_H >= min_H - 1.0, \
            f"H tổng {total_H:,.0f} < sàn {min_H:,.0f}"

    def test_allocation_matrix_shape(self, result_with_equity):
        """Ma trận phân bổ phải có shape 6×4."""
        mat = result_with_equity.allocation_matrix
        assert mat.shape == (6, 4), f"Shape sai: {mat.shape}"

    def test_allocation_by_region_sum_consistent(self, result_with_equity):
        """Tổng theo vùng + Tổng theo hạng mục phải bằng nhau."""
        by_r = result_with_equity.allocation_by_region.sum()
        by_j = result_with_equity.allocation_by_item.sum()
        assert abs(by_r - by_j) < 1.0, \
            f"Không nhất quán: by_region={by_r:.0f} ≠ by_item={by_j:.0f}"


# ─────────────────────────────────────────────────────────────
# Test 4: Ràng buộc công bằng vùng miền (C5)
# ─────────────────────────────────────────────────────────────

class TestEquityConstraint:
    """Kiểm tra ràng buộc công bằng vùng (C5 — Equity)."""

    def test_equity_reduces_objective(self, result_with_equity, result_no_equity):
        """Ràng buộc equity phải làm Z* giảm hoặc giữ nguyên."""
        assert result_no_equity.objective_value >= \
               result_with_equity.objective_value - 1.0, \
            "Lạ: có equity cho Z* cao hơn không equity"

    def test_equity_cost_nonneg(self, optimizer):
        """Chi phí equity (GDP gain mất đi) phải ≥ 0."""
        _, _, cost = optimizer.compute_equity_cost()
        assert cost >= -1.0, f"Chi phí equity âm: {cost}"

    def test_digital_index_ratio(self, result_with_equity, optimizer):
        """Sau đầu tư, digital index vùng yếu nhất / mạnh nhất ≥ λ."""
        lam = optimizer.lambda_eq
        dig = result_with_equity.digital_index_final
        if dig:
            max_d = max(dig.values())
            min_d = min(dig.values())
            if max_d > 0:
                ratio = min_d / max_d
                assert ratio >= lam - 0.05, \
                    f"Ratio {ratio:.3f} < lambda {lam} (±5% tolerance)"

    def test_no_equity_concentrates_wealth(self, result_with_equity, result_no_equity):
        """Không có equity → vùng giàu nhất được nhiều hơn."""
        by_r_eq   = result_with_equity.allocation_by_region
        by_r_no   = result_no_equity.allocation_by_region
        max_eq    = by_r_eq.max()
        max_no_eq = by_r_no.max()
        # Vùng nhận nhiều nhất trong no-equity ≥ vùng nhận nhiều nhất trong equity
        assert max_no_eq >= max_eq - 1.0


# ─────────────────────────────────────────────────────────────
# Test 5: So sánh kịch bản chính sách
# ─────────────────────────────────────────────────────────────

class TestScenarioSolve:
    """Kiểm tra solve_for_scenario cho 5 kịch bản."""

    @pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5"])
    def test_all_scenarios_feasible(self, optimizer, scenario_id):
        """Tất cả 5 kịch bản phải cho nghiệm khả thi."""
        result = optimizer.solve_for_scenario(scenario_id, with_equity=True)
        assert "Optimal" in result.status, \
            f"Kịch bản {scenario_id} không tối ưu: {result.status}"

    @pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5"])
    def test_scenario_result_type(self, optimizer, scenario_id):
        """solve_for_scenario phải trả về OptimizationResult."""
        result = optimizer.solve_for_scenario(scenario_id)
        assert isinstance(result, OptimizationResult)

    def test_ai_scenario_higher_ai_alloc(self, optimizer):
        """S3 (AI dẫn dắt) phải đầu tư AI nhiều hơn S4 (Bao trùm)."""
        r3 = optimizer.solve_for_scenario("S3")
        r4 = optimizer.solve_for_scenario("S4")
        ai_s3 = r3.allocation_matrix["AI"].sum()
        ai_s4 = r4.allocation_matrix["AI"].sum()
        assert ai_s3 > ai_s4, \
            f"S3 AI ({ai_s3:.0f}) ≤ S4 AI ({ai_s4:.0f})"

    def test_inclusive_scenario_higher_h_alloc(self, optimizer):
        """S4 (Bao trùm) phải đầu tư H nhiều hơn S3 (AI dẫn dắt)."""
        r3 = optimizer.solve_for_scenario("S3")
        r4 = optimizer.solve_for_scenario("S4")
        h_s3 = r3.allocation_matrix["H"].sum()
        h_s4 = r4.allocation_matrix["H"].sum()
        assert h_s4 > h_s3, \
            f"S4 H ({h_s4:.0f}) ≤ S3 H ({h_s3:.0f})"


# ─────────────────────────────────────────────────────────────
# Test 6: Phân tích độ nhạy ngân sách
# ─────────────────────────────────────────────────────────────

class TestSensitivityAnalysis:
    """Kiểm tra phân tích độ nhạy ngân sách."""

    def test_sensitivity_returns_dataframe(self, optimizer):
        """sensitivity_analysis_budget phải trả về DataFrame."""
        df = optimizer.sensitivity_analysis_budget(
            budget_range=[30_000, 40_000, 50_000]
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_sensitivity_z_monotone(self, optimizer):
        """Z* phải tăng (hoặc giữ nguyên) khi ngân sách tăng."""
        df = optimizer.sensitivity_analysis_budget(
            budget_range=[30_000, 40_000, 50_000, 60_000]
        )
        z = df["z_star"].dropna().values
        for i in range(len(z) - 1):
            assert z[i+1] >= z[i] - 1.0, \
                f"Z* giảm khi tăng ngân sách: {z[i]:.0f} → {z[i+1]:.0f}"

    def test_sensitivity_z_per_budget_positive(self, optimizer):
        """Hiệu quả biên Z*/B phải dương."""
        df = optimizer.sensitivity_analysis_budget(
            budget_range=[40_000, 50_000]
        )
        assert (df["z_per_budget"].dropna() > 0).all()

    def test_sensitivity_required_columns(self, optimizer):
        """DataFrame cần có đủ 3 cột."""
        df = optimizer.sensitivity_analysis_budget([40_000, 50_000])
        assert "budget_trillion" in df.columns
        assert "z_star"         in df.columns
        assert "z_per_budget"   in df.columns


# ─────────────────────────────────────────────────────────────
# Test 7: Đầu ra OptimizationResult
# ─────────────────────────────────────────────────────────────

class TestOptimizationResult:
    """Kiểm tra cấu trúc và tính nhất quán của OptimizationResult."""

    def test_summary_returns_string(self, result_with_equity):
        """summary() phải trả về string không rỗng."""
        s = result_with_equity.summary()
        assert isinstance(s, str) and len(s) > 50

    def test_allocation_by_region_index(self, result_with_equity):
        """allocation_by_region phải có index là mã vùng."""
        idx = set(result_with_equity.allocation_by_region.index)
        assert idx == set(REGIONS), f"Index sai: {idx}"

    def test_allocation_by_item_index(self, result_with_equity):
        """allocation_by_item phải có index là mã hạng mục."""
        idx = set(result_with_equity.allocation_by_item.index)
        assert idx == set(INVEST_ITEMS), f"Index sai: {idx}"

    def test_digital_index_final_all_regions(self, result_with_equity):
        """digital_index_final phải có giá trị cho tất cả 6 vùng."""
        dig = result_with_equity.digital_index_final
        assert len(dig) == 6

    def test_digital_index_improved(self, result_with_equity, optimizer):
        """Chỉ số số hóa sau đầu tư phải ≥ ban đầu."""
        for r, d_final in result_with_equity.digital_index_final.items():
            d_init = optimizer.D0.get(r, 0)
            assert d_final >= d_init - 1e-6, \
                f"Vùng {r}: D_final={d_final:.1f} < D_init={d_init}"

    def test_equity_cost_attribute(self, result_with_equity):
        """equity_cost mặc định = 0 (chưa tính)."""
        assert isinstance(result_with_equity.equity_cost, float)

    def test_scenario_id_stored(self, optimizer):
        """scenario_id phải được lưu trong kết quả."""
        for sid in ["S1", "S3"]:
            r = optimizer.solve_pulp(with_equity=True, scenario_id=sid)
            assert r.scenario_id == sid
