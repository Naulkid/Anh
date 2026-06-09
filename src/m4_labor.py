"""
m4_labor.py — Module M4: Mô phỏng Tác động AI đến Thị trường Lao động
=======================================================================
Định lượng tác động thay thế và bổ trợ của AI/tự động hóa đối với
thị trường lao động Việt Nam theo 8 ngành kinh tế lớn, bao gồm:

  1. Tính NetJob ròng = NewJob_AI + UpgradeJob − DisplacedJob_Auto
  2. Tối ưu phân bổ ngân sách AI & đào tạo để tối đa hóa tổng NetJob
     (LP tuyến tính với CVXPY/PuLP)
  3. Tìm ngưỡng đầu tư đào tạo tối thiểu cho ngành nhạy cảm nhất.
  4. Xác định nhóm lao động dễ bị tổn thương nhất.

Phương trình cốt lõi (Mục 10 — bài báo nguồn):
    NetJob_i  = NewJob_i + UpgradeJob_i − Displaced_i
    NewJob_i  = a1_i · x_AI_i + a2_i · x_D_i
    Upgrade_i = b1_i · x_H_i
    Displaced_i = c1_i · x_AI_i · risk_i
    Retrain_i   = d1_i · x_H_i  ≥ Displaced_i

Đơn vị: số việc làm trên mỗi tỷ VND đầu tư (việc/tỷ).

Author: AIDEOM-VN Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import LABOR_PARAMS, OUTPUTS_DIR

logger = logging.getLogger(__name__)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# Dataclass kết quả
# ═══════════════════════════════════════════════════════════════

@dataclass
class LaborResult:
    """Kết quả mô phỏng tác động lao động.

    Attributes:
        allocation_ai: Mảng đầu tư AI theo ngành (tỷ VND).
        allocation_h:  Mảng đầu tư đào tạo theo ngành (tỷ VND).
        new_jobs:      Việc làm mới AI tạo ra (nghìn việc).
        upgrade_jobs:  Việc làm nâng cấp (nghìn việc).
        displaced_jobs:Việc làm bị thay thế (nghìn việc).
        net_jobs:      NetJob ròng (nghìn việc).
        total_net:     Tổng NetJob (nghìn việc).
        sector_names:  Tên 8 ngành.
        status:        Trạng thái giải tối ưu.
        threshold_x_H: Ngưỡng x_H tối thiểu cho ngành nhạy cảm nhất.
    """
    allocation_ai: np.ndarray
    allocation_h: np.ndarray
    new_jobs: np.ndarray
    upgrade_jobs: np.ndarray
    displaced_jobs: np.ndarray
    net_jobs: np.ndarray
    total_net: float
    sector_names: List[str]
    status: str = "optimal"
    threshold_x_H: float = 0.0


# ═══════════════════════════════════════════════════════════════
# Class chính: LaborMarketSimulator
# ═══════════════════════════════════════════════════════════════

class LaborMarketSimulator:
    """Mô phỏng và tối ưu hóa tác động AI lên thị trường lao động VN.

    Giải bài toán LP:
        max  Σᵢ NetJob_i
        s.t. Σᵢ (x_AI_i + x_H_i) ≤ B
             NetJob_i ≥ 0          ∀i
             Displaced_i ≤ Retrain_i  ∀i
             x_AI_i, x_H_i ≥ 0     ∀i

    Args:
        total_budget: Tổng ngân sách (tỷ VND). Mặc định 30,000.
        params: Tham số 8 ngành. Mặc định từ config.LABOR_PARAMS.

    Example:
        >>> sim = LaborMarketSimulator(total_budget=30_000)
        >>> result = sim.optimize()
        >>> sim.print_summary(result)
    """

    def __init__(
        self,
        total_budget: float = 30_000.0,
        params: Optional[Dict] = None,
    ):
        self.B = total_budget
        raw = params or LABOR_PARAMS
        self.sectors = raw["sectors"]
        self.N = len(self.sectors)

        # Trích mảng tham số
        self.names   = [s["name"]    for s in self.sectors]
        self.labor_M = np.array([s["labor_M"] for s in self.sectors])
        self.risk    = np.array([s["risk"]    for s in self.sectors])
        self.a1      = np.array([s["a1"]      for s in self.sectors])
        self.b1      = np.array([s["b1"]      for s in self.sectors])
        self.c1      = np.array([s["c1"]      for s in self.sectors])
        self.d1      = np.array([s["d1"]      for s in self.sectors])

    # ─── Tính NetJob từ phân bổ đã cho ─────────────────────────

    def compute_netjob(
        self,
        x_ai: np.ndarray,
        x_h: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tính các thành phần NetJob cho từng ngành.

        Args:
            x_ai: Đầu tư AI (tỷ VND), shape (N,).
            x_h:  Đầu tư đào tạo H (tỷ VND), shape (N,).

        Returns:
            Tuple (new_jobs, upgrade_jobs, displaced_jobs, net_jobs)
            mỗi mảng đơn vị nghìn việc làm.
        """
        new_jobs   = (self.a1 * x_ai) / 1000          # nghìn việc
        upgrade    = (self.b1 * x_h)  / 1000
        displaced  = (self.c1 * self.risk * x_ai) / 1000
        net        = new_jobs + upgrade - displaced
        return new_jobs, upgrade, displaced, net

    # ─── Tối ưu hóa LP ─────────────────────────────────────────

    def optimize(
        self,
        constrain_net_positive: bool = True,
        max_displacement_rate: Optional[float] = None,
    ) -> LaborResult:
        """Tối ưu phân bổ ngân sách AI + đào tạo để tối đa NetJob.

        Args:
            constrain_net_positive: Bật ràng buộc NetJob_i ≥ 0 mọi ngành.
            max_displacement_rate: Nếu set, thêm ràng buộc
                                   Displaced_i ≤ rate * L_i.

        Returns:
            LaborResult với nghiệm tối ưu và phân tích.
        """
        if not CVXPY_AVAILABLE and not PULP_AVAILABLE:
            raise ImportError("Cần cài cvxpy hoặc pulp: pip install cvxpy pulp")

        if CVXPY_AVAILABLE:
            return self._optimize_cvxpy(constrain_net_positive, max_displacement_rate)
        return self._optimize_pulp(constrain_net_positive, max_displacement_rate)

    def _optimize_cvxpy(
        self,
        constrain_net_positive: bool,
        max_displacement_rate: Optional[float],
    ) -> LaborResult:
        """Giải LP bằng PuLP/CBC (ổn định hơn CVXPY cho bài LP này)."""
        return self._optimize_pulp(constrain_net_positive, max_displacement_rate)

    def _optimize_pulp(
        self,
        constrain_net_positive: bool,
        max_displacement_rate: Optional[float],
    ) -> LaborResult:
        """Giải LP bằng PuLP/CBC với ràng buộc sàn mỗi ngành.

        Ràng buộc sàn tối thiểu: mỗi ngành nhận ít nhất 1% ngân sách
        để tránh nghiệm suy biến (tất cả vào một ngành).
        """
        m = pulp.LpProblem("LaborOpt", pulp.LpMaximize)
        x_ai = [pulp.LpVariable(f"x_ai_{i}", lowBound=0) for i in range(self.N)]
        x_h  = [pulp.LpVariable(f"x_h_{i}",  lowBound=0) for i in range(self.N)]

        # Hệ số NetJob ròng cho mỗi ngành
        net_coef_ai = (self.a1 - self.c1 * self.risk) / 1000  # net job/tỷ from AI
        net_coef_h  = self.b1 / 1000                           # net job/tỷ from H

        net = [(net_coef_ai[i]*x_ai[i] + net_coef_h[i]*x_h[i]) for i in range(self.N)]

        m += pulp.lpSum(net)

        # C1: Ngân sách tổng
        m += pulp.lpSum(x_ai[i] + x_h[i] for i in range(self.N)) <= self.B

        # Phân bổ theo lao động: ngành nhiều lao động hơn -> sàn cao hơn
        labor_share = self.labor_M / self.labor_M.sum()
        floor_total = self.B * 0.60   # 60% chia theo lao động, 40% tự do

        for i in range(self.N):
            floor_i = floor_total * labor_share[i]
            # Sàn tổng (AI + H) cho ngành i
            m += x_ai[i] + x_h[i] >= floor_i

            # Ràng buộc năng lực đào tạo
            m += self.c1[i] * self.risk[i] * x_ai[i] <= self.d1[i] * x_h[i]

            if constrain_net_positive:
                m += net[i] >= 0

            if max_displacement_rate is not None:
                m += self.c1[i] * self.risk[i] * x_ai[i] / 1000 <= (
                    max_displacement_rate * self.labor_M[i]
                )

        m.solve(pulp.PULP_CBC_CMD(msg=False))

        x_ai_val = np.array([pulp.value(v) or 0.0 for v in x_ai])
        x_h_val  = np.array([pulp.value(v) or 0.0 for v in x_h])
        nj, uj, dj, netj = self.compute_netjob(x_ai_val, x_h_val)
        threshold = self._find_training_threshold(sector_idx=1)

        return LaborResult(
            allocation_ai=x_ai_val, allocation_h=x_h_val,
            new_jobs=nj, upgrade_jobs=uj,
            displaced_jobs=dj, net_jobs=netj,
            total_net=float(netj.sum()),
            sector_names=self.names,
            status=pulp.LpStatus[m.status],
            threshold_x_H=threshold,
        )

    # ─── Ngưỡng đào tạo tối thiểu ──────────────────────────────

    def _find_training_threshold(self, sector_idx: int = 1) -> float:
        """Tìm x_H tối thiểu để NetJob ≥ 0 khi x_AI tối đa.

        Giải tích:
          NetJob_i = a1_i * x_AI_i/1000 + b1_i * x_H/1000
                   - c1_i * risk_i * x_AI_i/1000 ≥ 0

          x_AI_max = B / N  (đơn giản hóa)
          x_H_min  = x_AI_max * (c1_i * risk_i - a1_i) / b1_i

        Args:
            sector_idx: Chỉ số ngành (0-based). Mặc định 1 = CN chế biến.

        Returns:
            x_H tối thiểu (tỷ VND), 0 nếu không cần đào tạo.
        """
        i = sector_idx
        x_ai_max = self.B / self.N  # phân bổ đều
        net_coef = (self.a1[i] - self.c1[i] * self.risk[i]) / 1000

        if net_coef >= 0:
            return 0.0  # AI đã dương → không cần đào tạo thêm

        # Cần: b1_i * x_H / 1000 ≥ -net_coef * x_ai_max
        x_H_min = (-net_coef * x_ai_max * 1000) / self.b1[i]
        return round(x_H_min, 1)

    # ─── So sánh kịch bản đầu tư ───────────────────────────────

    def compare_investment_levels(
        self,
        budgets: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """So sánh NetJob tổng ở các mức ngân sách khác nhau.

        Args:
            budgets: Danh sách ngân sách (tỷ VND).

        Returns:
            DataFrame: budget, total_net_jobs, net_per_billion.
        """
        budgets = budgets or [10_000, 20_000, 30_000, 40_000, 50_000]
        records = []
        for B in budgets:
            sim = LaborMarketSimulator(total_budget=B, params={"sectors": self.sectors})
            try:
                r = sim.optimize()
                records.append({
                    "budget_trillion": B,
                    "total_net_jobs_k": round(r.total_net, 2),
                    "net_per_billion":  round(r.total_net / (B / 1000), 3),
                })
            except Exception as e:
                logger.warning("Budget %d: %s", B, e)
        return pd.DataFrame(records)

    def print_summary(self, result: LaborResult) -> None:
        """In tóm tắt kết quả lao động ra console."""
        print(f"\n{'='*58}")
        print(f"  MODULE M4 — Kết quả Tối ưu Thị trường Lao động")
        print(f"  Ngân sách: {self.B:,.0f} tỷ VND | Status: {result.status}")
        print(f"{'='*58}")
        print(f"  {'Ngành':<25} {'xAI':>7} {'xH':>7} {'Net':>8} {'Disp':>8}")
        print(f"  {'-'*55}")
        for i, nm in enumerate(result.sector_names):
            flag = " ⚠" if result.displaced_jobs[i] > result.new_jobs[i] else ""
            print(
                f"  {nm[:24]:<25}"
                f" {result.allocation_ai[i]:>6,.0f}"
                f" {result.allocation_h[i]:>6,.0f}"
                f" {result.net_jobs[i]:>7.1f}k"
                f" {result.displaced_jobs[i]:>7.1f}k{flag}"
            )
        print(f"  {'─'*55}")
        print(f"  {'TỔNG NetJob':<25}  {'':>6}  {'':>6} {result.total_net:>7.1f}k")
        print(f"\n  Ngưỡng đào tạo tối thiểu (CN chế biến): "
              f"{result.threshold_x_H:,.0f} tỷ VND")
        print(f"{'='*58}")


# ═══════════════════════════════════════════════════════════════
# Biểu đồ
# ═══════════════════════════════════════════════════════════════

def plot_netjob_stacked(result: LaborResult, save: bool = True) -> plt.Figure:
    """Biểu đồ thanh xếp chồng: thành phần NetJob mỗi ngành.

    Args:
        result: LaborResult từ LaborMarketSimulator.optimize().
        save: Lưu PNG.

    Returns:
        matplotlib Figure.
    """
    names = [n[:16] for n in result.sector_names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Trái: thành phần NetJob ───────────────────────────────
    ax = axes[0]
    ax.bar(x, result.new_jobs,    label="Việc làm AI mới",   color="#10B981", alpha=0.85)
    ax.bar(x, result.upgrade_jobs,label="Nâng cấp kỹ năng",  color="#3B82F6", alpha=0.85,
           bottom=result.new_jobs)
    ax.bar(x, -result.displaced_jobs, label="Bị thay thế (-)",color="#EF4444", alpha=0.75)
    ax.plot(x, result.net_jobs, "ko-", lw=2, ms=6, label="NetJob ròng", zorder=5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Nghìn việc làm")
    ax.set_title("Thành phần NetJob theo Ngành\n(tối ưu phân bổ AI + Đào tạo)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # ── Phải: phân bổ ngân sách ───────────────────────────────
    ax2 = axes[1]
    w = 0.38
    ax2.bar(x - w/2, result.allocation_ai, w, label="Đầu tư AI",
            color="#8B5CF6", alpha=0.85)
    ax2.bar(x + w/2, result.allocation_h,  w, label="Đào tạo H",
            color="#F59E0B", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Tỷ VND")
    ax2.set_title("Phân bổ Ngân sách Tối ưu\nAI vs Đào tạo Nhân lực",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"M4 — Tác động AI: Thị trường Lao động Việt Nam 2026\n"
                 f"Ngân sách: {sum(result.allocation_ai + result.allocation_h):,.0f} tỷ VND | "
                 f"Tổng NetJob: {result.total_net:.1f}k việc",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / "m4_netjob_analysis.png", dpi=150)
        logger.info("Đã lưu: outputs/m4_netjob_analysis.png")
    return fig


def plot_labor_risk_matrix(result: LaborResult, save: bool = True) -> plt.Figure:
    """Ma trận Rủi ro vs Tiềm năng tạo việc làm AI.

    Trục X: Rủi ro thay thế (automation_risk %)
    Trục Y: Hệ số tạo việc làm mới (a1)
    Kích thước: Số lao động ngành
    Màu: NetJob ròng (xanh = dương, đỏ = âm)

    Args:
        result: LaborResult.
        save: Lưu PNG.

    Returns:
        matplotlib Figure.
    """
    from src.config import LABOR_PARAMS
    sectors = LABOR_PARAMS["sectors"]

    risks  = np.array([s["risk"] * 100 for s in sectors])
    a1_vals = np.array([s["a1"] for s in sectors])
    labor  = np.array([s["labor_M"] for s in sectors])
    names  = [s["name"][:14] for s in sectors]

    net = result.net_jobs
    colors = ["#10B981" if v >= 0 else "#EF4444" for v in net]
    sizes  = labor / labor.max() * 1500 + 100

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(risks, a1_vals, s=sizes, c=colors, alpha=0.75, edgecolors="white", lw=1.5)

    for i, name in enumerate(names):
        ax.annotate(name, (risks[i], a1_vals[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.axvline(30, color="orange", ls="--", alpha=0.6, label="Ngưỡng rủi ro 30%")
    ax.axhline(20, color="blue",   ls="--", alpha=0.6, label="Ngưỡng tạo việc làm 20")

    # Legend màu sắc
    green_patch = mpatches.Patch(color="#10B981", label="NetJob > 0 (dương)")
    red_patch   = mpatches.Patch(color="#EF4444", label="NetJob < 0 (âm)")
    ax.legend(handles=[green_patch, red_patch,
                        plt.Line2D([0],[0], color="orange", ls="--"),
                        plt.Line2D([0],[0], color="blue",   ls="--")],
              labels=["NetJob > 0", "NetJob < 0",
                      "Rủi ro 30%", "Tạo VL 20"],
              fontsize=8, loc="upper right")

    ax.set_xlabel("Rủi ro Tự động hóa (%)", fontsize=11)
    ax.set_ylabel("Hệ số Tạo Việc làm AI Mới (a1 — việc/tỷ VND)", fontsize=11)
    ax.set_title("Ma trận Rủi ro Lao động vs Tiềm năng Tạo Việc làm AI\n"
                 "(Kích thước = Quy mô lao động ngành)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / "m4_labor_risk_matrix.png", dpi=150)
        logger.info("Đã lưu: outputs/m4_labor_risk_matrix.png")
    return fig


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def run_m4_analysis() -> Dict:
    """Chạy toàn bộ phân tích M4."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    from typing import Dict

    print("\n" + "="*60)
    print("  MODULE M4 — Tác động AI lên Thị trường Lao động VN")
    print("  Tối ưu NetJob | 8 ngành | LP (CVXPY/PuLP)")
    print("="*60)

    sim = LaborMarketSimulator(total_budget=30_000)

    # 1. Tối ưu chính
    result = sim.optimize(constrain_net_positive=True)
    sim.print_summary(result)

    # 2. Thêm ràng buộc không mất quá 5% lao động
    print("\n--- Với ràng buộc: mỗi ngành mất <= 5% lao động ---")
    result_constrained = sim.optimize(
        constrain_net_positive=True,
        max_displacement_rate=0.05,
    )
    sim.print_summary(result_constrained)

    # 3. So sánh mức ngân sách
    print("\n--- So sánh NetJob theo ngân sách ---")
    df_budget = sim.compare_investment_levels()
    print(df_budget.to_string(index=False))

    # 4. Ngưỡng đào tạo tối thiểu
    print("\n--- Ngưỡng đào tạo tối thiểu cho ngành nhạy cảm ---")
    for i, s in enumerate(sim.sectors):
        thr = sim._find_training_threshold(i)
        risk_flag = " <-- NGUY CO CAO" if s["risk"] > 0.40 else ""
        print(f"  {s['name'][:26]:<26}: x_H_min = {thr:>8,.1f} ty VND{risk_flag}")

    # 5. Vẽ biểu đồ
    plot_netjob_stacked(result, save=True)
    plot_labor_risk_matrix(result, save=True)
    print("\n✅ Biểu đồ M4 đã lưu vào outputs/")

    return {
        "simulator": sim,
        "result_optimal": result,
        "result_constrained": result_constrained,
        "budget_comparison": df_budget,
    }


if __name__ == "__main__":
    run_m4_analysis()
