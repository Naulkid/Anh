"""
m3_optimization.py — Module M3: Tối ưu hóa Phân bổ Vốn Đầu tư
================================================================
Giải bài toán Quy hoạch Tuyến tính (LP) phân bổ ngân sách kinh tế số
quốc gia cho 6 vùng kinh tế-xã hội và 4 hạng mục đầu tư, đảm bảo:
  - Tối đa hóa GDP gain kỳ vọng (hàm mục tiêu)
  - Công bằng vùng miền (ràng buộc C5 – Equity Constraint)
  - Sàn/trần ngân sách mỗi vùng
  - Tỷ trọng tối thiểu nhân lực số

Mô hình:
    max  Z = Σᵣ Σⱼ β_{j,r} · x_{j,r}
    s.t. C1: Σ x_{j,r} ≤ B_total
         C2: Σⱼ x_{j,r} ≥ B_min   ∀r
         C3: Σⱼ x_{j,r} ≤ B_max   ∀r
         C4: Σᵣ x_{H,r} ≥ 0.24·B_total
         C5: D_r + γ·x_{D,r} ≥ λ·max_r(D_r + γ·x_{D,r})  [equity]
         C6: x_{j,r} ≥ 0

Thư viện: PuLP (solver CBC) + CVXPY để cross-validation.

Tài liệu tham chiếu:
    - QĐ 411/QĐ-TTg (2022) — Chiến lược Kinh tế số và Xã hội số
    - Bài tập 4: LP phân bổ ngân sách số theo ngành-vùng

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
import seaborn as sns

# ─── Thư viện tối ưu hóa ─────────────────────────────────────
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    logging.warning("PuLP không khả dụng. Cài: pip install pulp")

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logging.warning("CVXPY không khả dụng. Cài: pip install cvxpy")

# ─── import nội bộ ───────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    REGIONS, INVEST_ITEMS, BETA_MATRIX,
    DIGITAL_INDEX_INITIAL, BUDGET_CONSTRAINTS,
    REGION_NAMES_VI, ITEM_NAMES_VI, OUTPUTS_DIR, SCENARIOS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Dataclass kết quả tối ưu
# ═══════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """Kết quả đầu ra của Module M3.

    Attributes:
        status: Trạng thái giải ("Optimal", "Infeasible", ...).
        objective_value: Giá trị Z* (GDP gain tối ưu, tỷ VND).
        allocation_matrix: DataFrame [6 vùng × 4 hạng mục] — phân bổ tối ưu.
        allocation_by_region: Tổng đầu tư mỗi vùng.
        allocation_by_item: Tổng đầu tư mỗi hạng mục.
        shadow_prices: Giá đối ngẫu (dual values) của ràng buộc.
        digital_index_final: Chỉ số số hóa sau đầu tư mỗi vùng.
        solver_used: Tên solver sử dụng.
        equity_cost: Chi phí kinh tế của ràng buộc công bằng (tỷ VND GDP gain).
        scenario_id: Kịch bản chính sách tương ứng.
    """
    status: str
    objective_value: float
    allocation_matrix: pd.DataFrame
    allocation_by_region: pd.Series
    allocation_by_item: pd.Series
    shadow_prices: Dict[str, float] = field(default_factory=dict)
    digital_index_final: Dict[str, float] = field(default_factory=dict)
    solver_used: str = "PuLP/CBC"
    equity_cost: float = 0.0
    scenario_id: str = "S5"

    def summary(self) -> str:
        """In tóm tắt kết quả tối ưu."""
        lines = [
            f"\n{'='*55}",
            f"  KẾT QUẢ TỐI ƯU — Module M3 [{self.solver_used}]",
            f"{'='*55}",
            f"  Trạng thái    : {self.status}",
            f"  GDP gain Z*   : {self.objective_value:,.1f} tỷ VND",
            f"  Chi phí CK    : {self.equity_cost:,.1f} tỷ VND GDP gain",
            f"\n  Phân bổ theo vùng (tỷ VND):",
        ]
        for r, v in self.allocation_by_region.items():
            lines.append(f"    {REGION_NAMES_VI.get(r, r):35s}: {v:>8,.0f}")
        lines.append(f"\n  Phân bổ theo hạng mục (tỷ VND):")
        for j, v in self.allocation_by_item.items():
            lines.append(f"    {ITEM_NAMES_VI.get(j, j):20s}: {v:>8,.0f}")
        lines.append("="*55)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Class chính: BudgetOptimizer
# ═══════════════════════════════════════════════════════════════

class BudgetOptimizer:
    """Tối ưu hóa phân bổ ngân sách kinh tế số Việt Nam.

    Giải bài toán LP 24 biến (6 vùng × 4 hạng mục) với các ràng
    buộc ngân sách tổng, sàn/trần vùng, tỷ trọng nhân lực số và
    công bằng vùng miền (Equity Constraint - Mục 7.3 bài báo).

    Args:
        budget_constraints: Dict ràng buộc ngân sách. Mặc định từ config.
        beta_matrix: Dict hệ số tác động biên β_{j,r}. Mặc định từ config.
        digital_index_initial: Dict chỉ số số hóa ban đầu. Mặc định config.

    Example:
        >>> opt = BudgetOptimizer()
        >>> result = opt.solve(with_equity=True)
        >>> print(result.summary())
    """

    def __init__(
        self,
        budget_constraints: Optional[Dict] = None,
        beta_matrix: Optional[Dict] = None,
        digital_index_initial: Optional[Dict] = None,
    ):
        self.bc = budget_constraints or BUDGET_CONSTRAINTS.copy()
        self.beta = beta_matrix or BETA_MATRIX.copy()
        self.D0 = digital_index_initial or DIGITAL_INDEX_INITIAL.copy()

        self.regions = REGIONS
        self.items = INVEST_ITEMS

        # Tham số equity
        self.gamma_eq = self.bc["equity_gamma"]  # 0.002
        self.lambda_eq = self.bc["equity_lambda"]  # 0.70

    # ─── Solver chính: PuLP ────────────────────────────────────

    def solve_pulp(
        self,
        with_equity: bool = True,
        total_budget: Optional[float] = None,
        scenario_id: str = "S5",
    ) -> OptimizationResult:
        """Giải bài toán LP bằng PuLP với solver CBC.

        Args:
            with_equity: Bật/tắt ràng buộc công bằng vùng (C5).
            total_budget: Ghi đè tổng ngân sách (tỷ VND). Mặc định từ config.
            scenario_id: Mã kịch bản (ảnh hưởng metadata đầu ra).

        Returns:
            OptimizationResult với phân bổ tối ưu và metadata.

        Raises:
            ImportError: Nếu PuLP chưa được cài đặt.
            RuntimeError: Nếu bài toán vô nghiệm (Infeasible).
        """
        if not PULP_AVAILABLE:
            raise ImportError("Cài PuLP trước: pip install pulp")

        B = total_budget or self.bc["total_budget_trillion"]
        B_min = self.bc["min_per_region"]
        B_max = self.bc["max_per_region"]
        H_min_pct = self.bc["min_human_capital_pct"]

        # ── Khởi tạo mô hình ─────────────────────────────────
        m = pulp.LpProblem("AIDEOM_VN_Budget_Allocation", pulp.LpMaximize)

        # Biến quyết định x[r][j] ≥ 0
        x = pulp.LpVariable.dicts(
            "x",
            [(r, j) for r in self.regions for j in self.items],
            lowBound=0,
            cat="Continuous",
        )

        # ── Hàm mục tiêu: max Z = Σ β_{j,r} · x_{j,r} ──────
        m += pulp.lpSum(
            self.beta.get((r, j), 0) * x[(r, j)]
            for r in self.regions
            for j in self.items
        ), "Maximize_GDP_Gain"

        # ── C1: Tổng ngân sách ────────────────────────────────
        m += (
            pulp.lpSum(x[(r, j)] for r in self.regions for j in self.items) <= B,
            "C1_Total_Budget"
        )

        # ── C2 & C3: Sàn và trần mỗi vùng ────────────────────
        for r in self.regions:
            m += (
                pulp.lpSum(x[(r, j)] for j in self.items) >= B_min,
                f"C2_Floor_{r}"
            )
            m += (
                pulp.lpSum(x[(r, j)] for j in self.items) <= B_max,
                f"C3_Ceiling_{r}"
            )

        # ── C4: Sàn nhân lực số (≥24% tổng) ─────────────────
        m += (
            pulp.lpSum(x[(r, "H")] for r in self.regions) >= H_min_pct * B,
            "C4_Human_Capital_Floor"
        )

        # ── C5: Công bằng vùng (Equity Constraint) ───────────
        # D_r + γ·x_D,r ≥ λ·M   (tuyến tính hóa với biến phụ M)
        if with_equity:
            M_var = pulp.LpVariable("Dmax", lowBound=0)
            for r in self.regions:
                # D_r + γ·x_D,r ≤ M  → xác định M = max
                m += (
                    self.D0[r] + self.gamma_eq * x[(r, "D")] <= M_var,
                    f"C5a_DigMax_{r}"
                )
                # D_r + γ·x_D,r ≥ λ·M
                m += (
                    self.D0[r] + self.gamma_eq * x[(r, "D")] >= self.lambda_eq * M_var,
                    f"C5b_DigEq_{r}"
                )

        # ── Giải ─────────────────────────────────────────────
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
        status_code = m.solve(solver)
        status_str = pulp.LpStatus[m.status]

        if m.status != 1:  # 1 = Optimal
            logger.warning("⚠️  Bài toán không có nghiệm tối ưu: %s", status_str)

        # ── Trích xuất nghiệm ─────────────────────────────────
        alloc = {
            (r, j): pulp.value(x[(r, j)]) or 0.0
            for r in self.regions
            for j in self.items
        }
        z_star = pulp.value(m.objective) or 0.0

        # Ma trận phân bổ [vùng × hạng mục]
        alloc_mat = pd.DataFrame(
            {j: {r: alloc[(r, j)] for r in self.regions} for j in self.items}
        )
        alloc_mat.index.name = "region"

        # Tổng theo vùng & hạng mục
        by_region = alloc_mat.sum(axis=1).rename("total_invest")
        by_item   = alloc_mat.sum(axis=0).rename("total_invest")

        # Digital index sau đầu tư
        dig_final = {
            r: self.D0[r] + self.gamma_eq * alloc[(r, "D")]
            for r in self.regions
        }

        # Shadow prices (dual values) — thông qua sensitivity analysis
        shadow_prices = self._extract_shadow_prices_pulp(m)

        return OptimizationResult(
            status=status_str,
            objective_value=round(z_star, 2),
            allocation_matrix=alloc_mat,
            allocation_by_region=by_region,
            allocation_by_item=by_item,
            shadow_prices=shadow_prices,
            digital_index_final=dig_final,
            solver_used="PuLP/CBC",
            scenario_id=scenario_id,
        )

    def _extract_shadow_prices_pulp(
        self, model: "pulp.LpProblem"
    ) -> Dict[str, float]:
        """Trích xuất giá đối ngẫu (dual values) từ PuLP model.

        Args:
            model: LpProblem đã giải.

        Returns:
            Dict {constraint_name: dual_value}.
        """
        shadow = {}
        for name, constraint in model.constraints.items():
            try:
                # PuLP lưu dual trong .pi attribute sau khi giải
                dual = constraint.pi
                if dual is not None:
                    shadow[name] = round(float(dual), 4)
            except AttributeError:
                pass
        return shadow

    # ─── Solver phụ: CVXPY (cross-validation) ─────────────────

    def solve_cvxpy(
        self,
        with_equity: bool = True,
        total_budget: Optional[float] = None,
    ) -> OptimizationResult:
        """Giải bài toán LP bằng CVXPY (cross-validate với PuLP).

        Args:
            with_equity: Bật/tắt ràng buộc công bằng.
            total_budget: Ghi đè ngân sách tổng.

        Returns:
            OptimizationResult (solver_used = "CVXPY/ECOS").
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("Cài CVXPY trước: pip install cvxpy")

        B = total_budget or self.bc["total_budget_trillion"]
        n_r, n_j = len(self.regions), len(self.items)

        # Ma trận hệ số β (n_r × n_j)
        beta_mat = np.array([
            [self.beta.get((r, j), 0) for j in self.items]
            for r in self.regions
        ])

        # Biến quyết định
        X = cp.Variable((n_r, n_j), nonneg=True)

        # Hàm mục tiêu
        objective = cp.Maximize(cp.sum(cp.multiply(beta_mat, X)))
        constraints = []

        # C1: Ngân sách tổng
        constraints.append(cp.sum(X) <= B)

        # C2 & C3: Sàn/trần vùng
        for ri in range(n_r):
            constraints.append(cp.sum(X[ri, :]) >= self.bc["min_per_region"])
            constraints.append(cp.sum(X[ri, :]) <= self.bc["max_per_region"])

        # C4: Nhân lực số — H là cột cuối (index 3)
        H_idx = self.items.index("H")
        constraints.append(
            cp.sum(X[:, H_idx]) >= self.bc["min_human_capital_pct"] * B
        )

        # C5: Equity — tuyến tính hóa (chia nhỏ)
        if with_equity:
            D_idx = self.items.index("D")
            D0_vec = np.array([self.D0[r] for r in self.regions])
            dig_level = D0_vec + self.gamma_eq * X[:, D_idx]
            # Xấp xỉ: mỗi vùng ≥ λ * (mean digital level) — tuyến tính
            # (Approximation thay vì max để giữ LP)
            constraints.append(
                dig_level >= self.lambda_eq * cp.sum(dig_level) / n_r
            )

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)

        status_str = prob.status
        z_star = prob.value or 0.0

        if X.value is None:
            alloc_values = np.zeros((n_r, n_j))
        else:
            alloc_values = np.maximum(X.value, 0)

        alloc_mat = pd.DataFrame(
            alloc_values,
            index=self.regions,
            columns=self.items,
        )
        alloc_mat.index.name = "region"

        by_region = alloc_mat.sum(axis=1)
        by_item   = alloc_mat.sum(axis=0)

        dig_final = {}
        if X.value is not None:
            D_idx = self.items.index("D")
            for ri, r in enumerate(self.regions):
                dig_final[r] = self.D0[r] + self.gamma_eq * alloc_values[ri, D_idx]

        return OptimizationResult(
            status=status_str,
            objective_value=round(float(z_star), 2),
            allocation_matrix=alloc_mat,
            allocation_by_region=by_region,
            allocation_by_item=by_item,
            digital_index_final=dig_final,
            solver_used="CVXPY/ECOS",
        )

    # ─── Phân tích độ nhạy ngân sách ──────────────────────────

    def sensitivity_analysis_budget(
        self,
        budget_range: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Phân tích Z* khi thay đổi tổng ngân sách.

        Args:
            budget_range: Danh sách ngân sách (tỷ VND) để thử.
                          Mặc định: 30.000 đến 70.000 (bước 5.000).

        Returns:
            DataFrame với cột: budget, z_star, z_per_budget.
        """
        budgets = budget_range or list(range(30_000, 75_000, 5_000))
        records = []

        for B in budgets:
            try:
                res = self.solve_pulp(with_equity=True, total_budget=B)
                records.append({
                    "budget_trillion": B,
                    "z_star": res.objective_value,
                    "z_per_budget": res.objective_value / B,
                })
            except Exception as e:
                logger.warning("Budget %d: %s", B, e)
                records.append({
                    "budget_trillion": B,
                    "z_star": np.nan,
                    "z_per_budget": np.nan,
                })

        return pd.DataFrame(records)

    # ─── So sánh có/không ràng buộc công bằng ─────────────────

    def compute_equity_cost(
        self, total_budget: Optional[float] = None
    ) -> Tuple[OptimizationResult, OptimizationResult, float]:
        """Tính chi phí kinh tế của ràng buộc công bằng vùng.

        So sánh Z* của bài toán CÓ và KHÔNG có ràng buộc C5.

        Returns:
            Tuple: (result_with_equity, result_no_equity, cost_tỷ_VND).
        """
        B = total_budget or self.bc["total_budget_trillion"]
        res_with    = self.solve_pulp(with_equity=True,  total_budget=B)
        res_without = self.solve_pulp(with_equity=False, total_budget=B)

        cost = res_without.objective_value - res_with.objective_value
        res_with.equity_cost = round(cost, 2)

        logger.info(
            "Chi phí ràng buộc công bằng: %.2f tỷ VND GDP gain (%.2f%%)",
            cost, cost / max(res_without.objective_value, 1) * 100
        )
        return res_with, res_without, cost

    # ─── Tối ưu theo kịch bản chính sách ─────────────────────

    def solve_for_scenario(
        self,
        scenario_id: str,
        with_equity: bool = True,
    ) -> OptimizationResult:
        """Điều chỉnh hệ số β theo kịch bản và giải LP.

        Kịch bản S3 (AI dẫn dắt) nâng trọng số β_AI;
        S4 (Bao trùm) nâng β_H; S2 (Số hóa nhanh) nâng β_D.

        Args:
            scenario_id: Mã kịch bản ("S1"…"S5").
            with_equity: Bật ràng buộc công bằng vùng.

        Returns:
            OptimizationResult cho kịch bản đó.
        """
        scenario = SCENARIOS[scenario_id]
        alloc_w = scenario.allocation   # w_K, w_D, w_AI, w_H

        # Điều chỉnh β theo trọng số ưu tiên kịch bản
        weight_map = {"I": alloc_w["K"], "D": alloc_w["D"],
                      "AI": alloc_w["AI"], "H": alloc_w["H"]}
        # Scale β: β'_{j,r} = β_{j,r} * (w_j / 0.25)  [0.25 = equal baseline]
        adj_beta = {}
        for (r, j), val in self.beta.items():
            scale = weight_map.get(j, 0.25) / 0.25
            adj_beta[(r, j)] = val * scale

        # Tạm thời dùng beta điều chỉnh
        original_beta = self.beta
        self.beta = adj_beta
        result = self.solve_pulp(with_equity=with_equity, scenario_id=scenario_id)
        self.beta = original_beta   # khôi phục

        return result


# ═══════════════════════════════════════════════════════════════
# Hàm Vẽ biểu đồ
# ═══════════════════════════════════════════════════════════════

def plot_allocation_heatmap(
    result: OptimizationResult,
    save: bool = True,
) -> plt.Figure:
    """Vẽ heatmap phân bổ ngân sách tối ưu [vùng × hạng mục].

    Args:
        result: OptimizationResult từ BudgetOptimizer.
        save: Lưu file PNG.

    Returns:
        matplotlib Figure.
    """
    mat = result.allocation_matrix.copy()
    mat.index = [REGION_NAMES_VI.get(r, r) for r in mat.index]
    mat.columns = [ITEM_NAMES_VI.get(j, j) for j in mat.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        mat,
        annot=True, fmt=",.0f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Ngân sách (tỷ VND)"},
    )
    ax.set_title(
        f"Phân bổ Ngân sách Tối ưu theo Vùng-Hạng mục\n"
        f"(Z* = {result.objective_value:,.0f} tỷ VND GDP gain | {result.solver_used})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Hạng mục Đầu tư")
    ax.set_ylabel("Vùng Kinh tế - Xã hội")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / "m3_allocation_heatmap.png", dpi=150)
        logger.info("Đã lưu heatmap: outputs/m3_allocation_heatmap.png")

    return fig


def plot_scenario_comparison_m3(
    results: Dict[str, OptimizationResult],
    save: bool = True,
) -> plt.Figure:
    """Vẽ biểu đồ thanh so sánh Z* của 5 kịch bản.

    Args:
        results: Dict {scenario_id: OptimizationResult}.
        save: Lưu file PNG.

    Returns:
        matplotlib Figure.
    """
    sids = list(results.keys())
    names = [SCENARIOS[s].name_vi for s in sids]
    z_vals = [results[s].objective_value for s in sids]
    colors_map = {"S1": "#6B7280", "S2": "#3B82F6",
                  "S3": "#8B5CF6", "S4": "#10B981", "S5": "#F59E0B"}

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        names, z_vals,
        color=[colors_map.get(s, "gray") for s in sids],
        edgecolor="white", linewidth=1.5,
    )
    for bar, val in zip(bars, z_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(z_vals) * 0.005,
            f"{val:,.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_title(
        "So sánh GDP Gain tối ưu Z* theo 5 Kịch bản Chính sách\n"
        "(Ngân sách 50.000 tỷ VND | Có ràng buộc công bằng vùng)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Z* — GDP gain (tỷ VND)")
    ax.set_ylim(0, max(z_vals) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / "m3_scenario_comparison.png", dpi=150)

    return fig


def plot_sensitivity_budget(
    df_sens: pd.DataFrame, save: bool = True
) -> plt.Figure:
    """Vẽ đường cong Z*(B) — phân tích độ nhạy ngân sách.

    Args:
        df_sens: DataFrame từ BudgetOptimizer.sensitivity_analysis_budget().
        save: Lưu file PNG.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(df_sens["budget_trillion"], df_sens["z_star"],
             "bo-", lw=2, ms=7)
    ax1.set_title("Đường cong Z*(B) — GDP gain theo Ngân sách")
    ax1.set_xlabel("Ngân sách tổng (tỷ VND)")
    ax1.set_ylabel("Z* (tỷ VND GDP gain)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_sens["budget_trillion"], df_sens["z_per_budget"],
             "rs-", lw=2, ms=7)
    ax2.set_title("Hiệu quả Biên — GDP gain / tỷ VND đầu tư")
    ax2.set_xlabel("Ngân sách tổng (tỷ VND)")
    ax2.set_ylabel("Z*/B (tỷ VND GDP / tỷ VND đầu tư)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Phân tích Độ nhạy: Tác động Ngân sách đến Kết quả Tối ưu",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / "m3_sensitivity_budget.png", dpi=150)

    return fig


# ═══════════════════════════════════════════════════════════════
# Entry point chạy độc lập
# ═══════════════════════════════════════════════════════════════

def run_m3_analysis() -> Dict:
    """Chạy toàn bộ phân tích M3 và trả kết quả cho pipeline.

    Returns:
        Dict chứa: optimizer, result_optimal, results_by_scenario,
        sensitivity_df, equity_cost.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("\n" + "="*60)
    print("  MODULE M3 — Tối ưu hóa Phân bổ Vốn Kinh tế số")
    print("  Giải LP 24 biến | 6 vùng × 4 hạng mục")
    print("="*60)

    opt = BudgetOptimizer()

    # 1. Giải tối ưu chính (S5 - Cân bằng) với equity
    print("\n🔧 Giải LP kịch bản S5 (Cân bằng tối ưu) — PuLP/CBC...")
    result_optimal = opt.solve_pulp(with_equity=True, scenario_id="S5")
    print(result_optimal.summary())

    # 2. Cross-validate bằng CVXPY
    if CVXPY_AVAILABLE:
        print("\n🔧 Cross-validate bằng CVXPY...")
        result_cvxpy = opt.solve_cvxpy(with_equity=True)
        diff = abs(result_optimal.objective_value - result_cvxpy.objective_value)
        status = "✅ Nhất quán" if diff < 1.0 else f"⚠️ Lệch {diff:.2f}"
        print(f"   PuLP Z*  = {result_optimal.objective_value:,.2f}")
        print(f"   CVXPY Z* = {result_cvxpy.objective_value:,.2f}")
        print(f"   Kết quả: {status}")

    # 3. Chi phí công bằng vùng miền
    print("\n📊 Tính chi phí ràng buộc công bằng...")
    r_with, r_without, eq_cost = opt.compute_equity_cost()
    print(f"   Z* (có equity):   {r_with.objective_value:>10,.2f} tỷ VND")
    print(f"   Z* (không equity): {r_without.objective_value:>10,.2f} tỷ VND")
    print(f"   Chi phí kinh tế:  {eq_cost:>10,.2f} tỷ VND GDP gain")
    pct = eq_cost / max(r_without.objective_value, 1) * 100
    print(f"   (~{pct:.1f}% — giá phải trả cho công bằng vùng miền)")

    # 4. Giải theo tất cả kịch bản
    print("\n📊 Giải LP theo 5 kịch bản chính sách...")
    results_by_scenario = {}
    for sid in ["S1", "S2", "S3", "S4", "S5"]:
        r = opt.solve_for_scenario(sid, with_equity=True)
        results_by_scenario[sid] = r
        print(f"   {sid}: Z* = {r.objective_value:>10,.2f} tỷ VND")

    # 5. Phân tích độ nhạy ngân sách
    print("\n📊 Phân tích độ nhạy ngân sách (30k–70k tỷ)...")
    sensitivity_df = opt.sensitivity_analysis_budget()
    print(sensitivity_df.to_string(index=False))

    # 6. Ma trận phân bổ tối ưu
    print("\n📋 Ma trận phân bổ tối ưu (S5):")
    mat = result_optimal.allocation_matrix.copy()
    mat.index = [REGION_NAMES_VI.get(r, r) for r in mat.index]
    mat.columns = [ITEM_NAMES_VI.get(j, j) for j in mat.columns]
    mat["TỔNG"] = mat.sum(axis=1)
    print(mat.round(0).to_string())

    # 7. Vẽ biểu đồ
    plot_allocation_heatmap(result_optimal, save=True)
    plot_scenario_comparison_m3(results_by_scenario, save=True)
    plot_sensitivity_budget(sensitivity_df, save=True)
    print("\n✅ Biểu đồ M3 đã lưu vào outputs/")

    return {
        "optimizer": opt,
        "result_optimal": result_optimal,
        "results_by_scenario": results_by_scenario,
        "sensitivity_df": sensitivity_df,
        "equity_cost": eq_cost,
    }


if __name__ == "__main__":
    run_m3_analysis()


# ═══════════════════════════════════════════════════════════════
# Pyomo Solver — theo yêu cầu đề bài (Bài 10, Bài 4)
# ═══════════════════════════════════════════════════════════════

class PyomoBudgetOptimizer:
    """Tối ưu phân bổ ngân sách bằng Pyomo + GLPK/CBC.

    Triển khai cùng mô hình LP như BudgetOptimizer nhưng dùng
    Pyomo — thư viện mô hình hóa đại số tối ưu hóa (AML) chuẩn
    học thuật, tương thích nhiều solver (GLPK, CBC, Gurobi, CPLEX).

    Ưu điểm so với PuLP:
      - Cú pháp khai báo ràng buộc tường minh hơn (Set, Param, Var)
      - Hỗ trợ stochastic programming (Bài 10)
      - Tích hợp tốt với Pyomo.DAE cho bài toán động (Bài 8)

    Args:
        solver_name: "glpk" (mặc định, miễn phí) hoặc "cbc", "gurobi".
        budget_constraints: Ràng buộc ngân sách từ config.
        beta_matrix: Hệ số tác động biên β_{j,r}.
        digital_index_initial: Chỉ số số hóa ban đầu D₀_r.

    Example:
        >>> opt = PyomoBudgetOptimizer(solver_name="glpk")
        >>> result = opt.solve(with_equity=True)
        >>> print(result.summary())
    """

    def __init__(
        self,
        solver_name: str = "glpk",
        budget_constraints: Optional[Dict] = None,
        beta_matrix: Optional[Dict] = None,
        digital_index_initial: Optional[Dict] = None,
    ):
        try:
            import pyomo.environ as pyo
            self._pyo = pyo
        except ImportError:
            raise ImportError("Cài Pyomo: pip install pyomo")

        self.solver_name = solver_name
        self.bc  = budget_constraints or BUDGET_CONSTRAINTS.copy()
        self.beta = beta_matrix or BETA_MATRIX.copy()
        self.D0   = digital_index_initial or DIGITAL_INDEX_INITIAL.copy()
        self.regions = REGIONS
        self.items   = INVEST_ITEMS
        self.gamma_eq  = self.bc["equity_gamma"]
        self.lambda_eq = self.bc["equity_lambda"]

    def solve(
        self,
        with_equity: bool = True,
        total_budget: Optional[float] = None,
        scenario_id: str = "S5",
    ) -> "OptimizationResult":
        """Giải LP bằng Pyomo với cấu trúc Set/Param/Var chuẩn học thuật.

        Mô hình:
            Sets  : J (hạng mục), R (vùng)
            Params: beta[J,R], D0[R], B, B_min, B_max, gamma, lambda_eq
            Vars  : x[J,R] ≥ 0, M ≥ 0 (auxiliary cho equity)
            Obj   : max Σ_j Σ_r beta[j,r] * x[j,r]
            C1    : Σ x[j,r] ≤ B
            C2    : Σ_j x[j,r] ≥ B_min   ∀r
            C3    : Σ_j x[j,r] ≤ B_max   ∀r
            C4    : Σ_r x['H',r] ≥ 0.24*B
            C5a   : D0[r] + γ*x['D',r] ≤ M   ∀r  (define M=max)
            C5b   : D0[r] + γ*x['D',r] ≥ λ*M ∀r  (equity floor)

        Args:
            with_equity: Bật/tắt ràng buộc công bằng vùng (C5).
            total_budget: Ghi đè tổng ngân sách. Mặc định từ config.
            scenario_id: Mã kịch bản (lưu vào metadata đầu ra).

        Returns:
            OptimizationResult — cùng cấu trúc với PuLP solver.

        Raises:
            RuntimeError: Nếu solver không khả dụng hoặc bài toán vô nghiệm.
        """
        pyo = self._pyo
        B      = total_budget or self.bc["total_budget_trillion"]
        B_min  = self.bc["min_per_region"]
        B_max  = self.bc["max_per_region"]
        H_min  = self.bc["min_human_capital_pct"] * B

        # ── Kiểm tra solver ──────────────────────────────────
        solver = pyo.SolverFactory(self.solver_name)
        if not solver.available():
            raise RuntimeError(
                f"Solver '{self.solver_name}' không khả dụng. "
                f"Cài GLPK: apt install glpk-utils | hoặc dùng 'cbc'"
            )

        # ── Khởi tạo ConcreteModel ───────────────────────────
        m = pyo.ConcreteModel(name="AIDEOM_VN_Pyomo")

        # Sets
        m.R = pyo.Set(initialize=self.regions)
        m.J = pyo.Set(initialize=self.items)

        # Parameters
        m.beta  = pyo.Param(m.R, m.J,
                            initialize={(r, j): self.beta.get((r, j), 0.0)
                                        for r in self.regions for j in self.items})
        m.D0    = pyo.Param(m.R, initialize=self.D0)
        m.B     = pyo.Param(initialize=B)
        m.B_min = pyo.Param(initialize=B_min)
        m.B_max = pyo.Param(initialize=B_max)
        m.H_min = pyo.Param(initialize=H_min)
        m.gamma = pyo.Param(initialize=self.gamma_eq)
        m.lam   = pyo.Param(initialize=self.lambda_eq)

        # Decision variables
        m.x = pyo.Var(m.R, m.J, within=pyo.NonNegativeReals)
        if with_equity:
            m.Dmax = pyo.Var(within=pyo.NonNegativeReals)

        # Objective: max GDP gain
        m.obj = pyo.Objective(
            expr=sum(m.beta[r, j] * m.x[r, j]
                     for r in m.R for j in m.J),
            sense=pyo.maximize,
        )

        # C1: Tổng ngân sách
        m.c1 = pyo.Constraint(
            expr=sum(m.x[r, j] for r in m.R for j in m.J) <= m.B
        )

        # C2: Sàn mỗi vùng
        m.c2 = pyo.Constraint(m.R, rule=lambda mod, r:
            sum(mod.x[r, j] for j in mod.J) >= mod.B_min)

        # C3: Trần mỗi vùng
        m.c3 = pyo.Constraint(m.R, rule=lambda mod, r:
            sum(mod.x[r, j] for j in mod.J) <= mod.B_max)

        # C4: Nhân lực số tối thiểu
        m.c4 = pyo.Constraint(
            expr=sum(m.x[r, "H"] for r in m.R) >= m.H_min
        )

        # C5: Equity (tuyến tính hóa với biến phụ Dmax)
        if with_equity:
            m.c5a = pyo.Constraint(m.R, rule=lambda mod, r:
                mod.D0[r] + mod.gamma * mod.x[r, "D"] <= mod.Dmax)
            m.c5b = pyo.Constraint(m.R, rule=lambda mod, r:
                mod.D0[r] + mod.gamma * mod.x[r, "D"] >= mod.lam * mod.Dmax)

        # ── Giải ──────────────────────────────────────────────
        results = solver.solve(m, tee=False)
        term = str(results.solver.termination_condition)
        status_str = "Optimal" if "optimal" in term.lower() else term

        # ── Trích xuất nghiệm ─────────────────────────────────
        alloc = {
            (r, j): pyo.value(m.x[r, j]) or 0.0
            for r in self.regions for j in self.items
        }
        z_star = pyo.value(m.obj) or 0.0

        alloc_mat = pd.DataFrame(
            {j: {r: alloc[(r, j)] for r in self.regions}
             for j in self.items}
        )
        alloc_mat.index.name = "region"

        dig_final = {
            r: self.D0[r] + self.gamma_eq * alloc[(r, "D")]
            for r in self.regions
        }

        logger.info(
            "Pyomo/%s: status=%s | Z*=%.2f",
            self.solver_name, status_str, z_star
        )

        return OptimizationResult(
            status=status_str,
            objective_value=round(z_star, 2),
            allocation_matrix=alloc_mat,
            allocation_by_region=alloc_mat.sum(axis=1),
            allocation_by_item=alloc_mat.sum(axis=0),
            digital_index_final=dig_final,
            solver_used=f"Pyomo/{self.solver_name.upper()}",
            scenario_id=scenario_id,
        )

    def cross_validate_with_pulp(
        self,
        with_equity: bool = True,
    ) -> Dict:
        """So sánh kết quả Pyomo vs PuLP để kiểm tra nhất quán.

        Returns:
            Dict: {pyomo_z, pulp_z, diff_abs, diff_pct, consistent}.
        """
        r_pyomo = self.solve(with_equity=with_equity)
        pulp_opt = BudgetOptimizer(
            budget_constraints=self.bc,
            beta_matrix=self.beta,
            digital_index_initial=self.D0,
        )
        r_pulp = pulp_opt.solve_pulp(with_equity=with_equity)

        diff_abs = abs(r_pyomo.objective_value - r_pulp.objective_value)
        diff_pct = diff_abs / max(r_pulp.objective_value, 1) * 100
        consistent = diff_pct < 0.5  # < 0.5% coi là nhất quán

        logger.info(
            "Cross-validate: Pyomo=%.2f | PuLP=%.2f | diff=%.2f (%.3f%%)",
            r_pyomo.objective_value, r_pulp.objective_value,
            diff_abs, diff_pct,
        )
        return {
            "pyomo_z":    r_pyomo.objective_value,
            "pulp_z":     r_pulp.objective_value,
            "diff_abs":   round(diff_abs, 2),
            "diff_pct":   round(diff_pct, 4),
            "consistent": consistent,
            "pyomo_result": r_pyomo,
            "pulp_result":  r_pulp,
        }
