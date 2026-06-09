"""
m2_readiness.py — Module M2: Đánh giá Sẵn sàng AI/Số
=======================================================
Xếp hạng 6 vùng kinh tế-xã hội và 10 ngành theo mức độ sẵn sàng
triển khai AI và chuyển đổi số, dùng hai phương pháp:

  1. TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
     với trọng số chuyên gia.
  2. Entropy Weight Method — xác định trọng số khách quan từ phân tán dữ liệu.
  3. Phân tích độ nhạy trọng số AI Readiness.

Quy trình TOPSIS (Bài 6 — đề bài):
  B1: Chuẩn hóa vector  r_ij = x_ij / sqrt(Σ x_ij²)
  B2: Ma trận có trọng số  v_ij = w_j * r_ij
  B3: Lời giải lý tưởng A⁺/A⁻
  B4: Khoảng cách Euclide S⁺, S⁻
  B5: Hệ số gần gũi C* = S⁻ / (S⁺ + S⁻)

Tài liệu tham chiếu:
    Hwang & Yoon (1981). Multiple Attribute Decision Making.
    QĐ 127/QĐ-TTg (2021) — Chiến lược AI quốc gia đến 2030.

Author: AIDEOM-VN Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_regions, load_sectors, normalize_minmax
from src.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Dataclass kết quả
# ═══════════════════════════════════════════════════════════════

@dataclass
class ReadinessResult:
    """Kết quả xếp hạng sẵn sàng AI/Số.

    Attributes:
        entity_type: "region" hoặc "sector".
        scores_expert: Series C* với trọng số chuyên gia.
        scores_entropy: Series C* với trọng số Entropy.
        weights_expert: Array trọng số chuyên gia.
        weights_entropy: Array trọng số Entropy.
        criteria_names: Danh sách tên tiêu chí.
        ranking_expert: DataFrame xếp hạng theo trọng số chuyên gia.
        ranking_entropy: DataFrame xếp hạng theo trọng số Entropy.
        normalized_matrix: Ma trận đã chuẩn hóa vector.
        top3_stable: bool — top-3 có thay đổi giữa hai bộ trọng số không.
    """
    entity_type: str
    scores_expert: pd.Series
    scores_entropy: pd.Series
    weights_expert: np.ndarray
    weights_entropy: np.ndarray
    criteria_names: List[str]
    ranking_expert: pd.DataFrame
    ranking_entropy: pd.DataFrame
    normalized_matrix: np.ndarray
    top3_stable: bool = True


# ═══════════════════════════════════════════════════════════════
# Class chính: TOPSISRanker
# ═══════════════════════════════════════════════════════════════

class TOPSISRanker:
    """Triển khai TOPSIS + Entropy Weight cho xếp hạng sẵn sàng AI.

    Args:
        entity_type: "region" (6 vùng) hoặc "sector" (10 ngành).

    Example:
        >>> ranker = TOPSISRanker(entity_type="region")
        >>> result = ranker.rank()
        >>> print(result.ranking_expert)
    """

    # Tiêu chí cho vùng kinh tế-xã hội
    REGION_CRITERIA = [
        "grdp_per_capita_million_VND",   # benefit
        "fdi_registered_billion_USD",     # benefit
        "digital_index_0_100",            # benefit
        "ai_readiness_0_100",             # benefit
        "trained_labor_pct",              # benefit
        "rd_intensity_pct",               # benefit
        "internet_penetration_pct",       # benefit
        "gini_coef",                      # cost  (càng thấp càng tốt)
    ]
    REGION_IS_BENEFIT = [True, True, True, True, True, True, True, False]
    REGION_LABELS = [
        "GRDP/người", "FDI", "Số hóa", "AI Readiness",
        "LĐ đào tạo", "R&D/GRDP", "Internet", "Gini (−)"
    ]
    # Trọng số chuyên gia cho vùng (Bài 6 đề bài)
    REGION_WEIGHTS_EXPERT = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

    # Tiêu chí cho ngành kinh tế
    SECTOR_CRITERIA = [
        "growth_rate_2024_pct",    # benefit
        "productivity_million_VND",# benefit
        "spillover_coef_0_1",      # benefit
        "export_billion_USD",      # benefit
        "labor_million",           # benefit
        "ai_readiness_0_100",      # benefit
        "automation_risk_pct",     # cost
    ]
    SECTOR_IS_BENEFIT = [True, True, True, True, True, True, False]
    SECTOR_LABELS = [
        "Tăng trưởng", "Năng suất", "Lan tỏa", "Xuất khẩu",
        "Việc làm", "AI Readiness", "Rủi ro TĐH (−)"
    ]
    # Trọng số chuyên gia cho ngành (Bài 3 đề bài)
    SECTOR_WEIGHTS_EXPERT = np.array([0.136386, 0.136386, 0.181818, 0.136386, 0.090909, 0.181818, 0.136386])

    def __init__(self, entity_type: str = "region"):
        if entity_type not in ("region", "sector"):
            raise ValueError("entity_type phải là 'region' hoặc 'sector'")
        self.entity_type = entity_type
        self._setup_config()

    def _setup_config(self) -> None:
        """Thiết lập cấu hình tiêu chí theo loại đối tượng."""
        if self.entity_type == "region":
            self.criteria    = self.REGION_CRITERIA
            self.is_benefit  = self.REGION_IS_BENEFIT
            self.labels      = self.REGION_LABELS
            self.w_expert    = self.REGION_WEIGHTS_EXPERT.copy()
        else:
            self.criteria    = self.SECTOR_CRITERIA
            self.is_benefit  = self.SECTOR_IS_BENEFIT
            self.labels      = self.SECTOR_LABELS
            self.w_expert    = self.SECTOR_WEIGHTS_EXPERT.copy()

        assert abs(self.w_expert.sum() - 1.0) < 1e-4, \
            f"Trọng số chuyên gia không tổng = 1: {self.w_expert.sum()}"

    # ─── Load dữ liệu ──────────────────────────────────────────

    def _load_data(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Nạp dữ liệu và trả về ma trận X, nhãn thực thể."""
        if self.entity_type == "region":
            df = load_regions()
            names = df["region_name_vi"].tolist()
        else:
            df = load_sectors()
            names = df["sector_name_vi"].tolist()

        X = df[self.criteria].values.astype(float)
        return df, X, names

    # ─── TOPSIS core ───────────────────────────────────────────

    def _topsis(
        self, X: np.ndarray, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tính TOPSIS thuần từ ma trận X và trọng số w.

        Args:
            X: Ma trận quyết định (n_entities × n_criteria).
            weights: Mảng trọng số (n_criteria,). Phải tổng = 1.

        Returns:
            Tuple: (C_star, S_plus, S_minus)
              C_star  — hệ số gần gũi [0,1], càng cao càng tốt.
              S_plus  — khoảng cách đến lý tưởng dương.
              S_minus — khoảng cách đến lý tưởng âm.
        """
        n, m = X.shape

        # B1: Chuẩn hóa vector
        col_norms = np.sqrt((X ** 2).sum(axis=0))
        col_norms[col_norms == 0] = 1e-12
        R = X / col_norms

        # B2: Ma trận có trọng số
        V = R * weights

        # B3: Lời giải lý tưởng A⁺ và A⁻
        A_plus  = np.where(self.is_benefit, V.max(axis=0), V.min(axis=0))
        A_minus = np.where(self.is_benefit, V.min(axis=0), V.max(axis=0))

        # B4: Khoảng cách Euclide
        S_plus  = np.sqrt(((V - A_plus)  ** 2).sum(axis=1))
        S_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))

        # B5: Hệ số gần gũi tương đối
        denom = S_plus + S_minus
        denom[denom == 0] = 1e-12
        C_star = S_minus / denom

        return C_star, S_plus, S_minus

    # ─── Entropy Weight ────────────────────────────────────────

    def entropy_weights(self, X: np.ndarray) -> np.ndarray:
        """Tính trọng số khách quan bằng phương pháp Entropy.

        Phương pháp: Tiêu chí có phân tán thông tin lớn hơn →
        đóng góp nhiều hơn vào phân biệt các phương án → trọng số cao hơn.

        Công thức:
          P_ij = x_ij / Σᵢ x_ij
          E_j  = -(1/ln n) · Σᵢ P_ij · ln(P_ij)
          d_j  = 1 - E_j
          w_j  = d_j / Σⱼ d_j

        Args:
            X: Ma trận dữ liệu gốc (n × m), tất cả giá trị > 0.

        Returns:
            Mảng trọng số Entropy (m,), tổng = 1.
        """
        # Đảm bảo giá trị dương (shift nếu có âm)
        X_pos = X - X.min(axis=0) + 1e-6

        n = X_pos.shape[0]
        P = X_pos / X_pos.sum(axis=0)

        k = 1.0 / np.log(n + 1e-12)
        # Tránh log(0)
        log_P = np.where(P > 0, np.log(P + 1e-12), 0.0)
        E = -k * (P * log_P).sum(axis=0)

        d = 1.0 - E
        d = np.maximum(d, 0)  # clip âm do số học
        w_total = d.sum()
        if w_total < 1e-12:
            return np.ones(X.shape[1]) / X.shape[1]
        return d / w_total

    # ─── Hàm chính: rank ───────────────────────────────────────

    def rank(self) -> ReadinessResult:
        """Xếp hạng các đối tượng theo TOPSIS với 2 bộ trọng số.

        Returns:
            ReadinessResult chứa xếp hạng, điểm, trọng số.
        """
        df, X, names = self._load_data()

        # ── Trọng số chuyên gia ──────────────────────────────
        c_expert, _, _ = self._topsis(X, self.w_expert)

        # ── Trọng số Entropy ─────────────────────────────────
        w_entropy = self.entropy_weights(X)
        c_entropy, _, _ = self._topsis(X, w_entropy)

        # ── Chuẩn hóa vector (lưu để plot) ───────────────────
        col_norms = np.sqrt((X ** 2).sum(axis=0))
        col_norms[col_norms == 0] = 1e-12
        R_norm = X / col_norms

        # ── Tạo DataFrame xếp hạng ───────────────────────────
        rank_exp = pd.DataFrame({
            "name": names,
            "topsis_score": np.round(c_expert, 4),
        }).sort_values("topsis_score", ascending=False).reset_index(drop=True)
        rank_exp["rank"] = rank_exp.index + 1

        rank_ent = pd.DataFrame({
            "name": names,
            "topsis_score": np.round(c_entropy, 4),
        }).sort_values("topsis_score", ascending=False).reset_index(drop=True)
        rank_ent["rank"] = rank_ent.index + 1

        # Top-3 ổn định?
        top3_exp = set(rank_exp.head(3)["name"])
        top3_ent = set(rank_ent.head(3)["name"])
        top3_stable = top3_exp == top3_ent

        return ReadinessResult(
            entity_type=self.entity_type,
            scores_expert=pd.Series(c_expert, index=names),
            scores_entropy=pd.Series(c_entropy, index=names),
            weights_expert=self.w_expert,
            weights_entropy=w_entropy,
            criteria_names=self.labels,
            ranking_expert=rank_exp,
            ranking_entropy=rank_ent,
            normalized_matrix=R_norm,
            top3_stable=top3_stable,
        )

    # ─── Phân tích độ nhạy trọng số AI Readiness ───────────────

    def sensitivity_ai_weight(self) -> pd.DataFrame:
        """Phân tích độ nhạy: thay đổi w_AI từ 0.05 đến 0.40.

        Tăng dần w_AI, chuẩn hóa lại tổng = 1 bằng cách thu hẹp
        các trọng số khác tỷ lệ đều nhau. Ghi nhận top-3 mỗi bước.

        Returns:
            DataFrame: w_ai, rank_1, rank_2, rank_3, all_scores...
        """
        df, X, names = self._load_data()

        # Xác định index tiêu chí AI Readiness
        ai_label = "ai_readiness_0_100"
        try:
            ai_idx = self.criteria.index(ai_label)
        except ValueError:
            logger.warning("Không tìm thấy cột ai_readiness_0_100")
            return pd.DataFrame()

        w_ai_range = np.arange(0.05, 0.45, 0.05)
        records = []

        for w_ai_val in w_ai_range:
            # Trọng số mới: AI = w_ai_val, còn lại scale proportionally
            w_new = self.w_expert.copy()
            other_sum = 1.0 - w_ai_val
            other_indices = [i for i in range(len(w_new)) if i != ai_idx]
            old_other_sum = w_new[other_indices].sum()

            if old_other_sum > 1e-10:
                for i in other_indices:
                    w_new[i] = w_new[i] / old_other_sum * other_sum
            w_new[ai_idx] = w_ai_val

            c_star, _, _ = self._topsis(X, w_new)
            ranked = sorted(zip(names, c_star), key=lambda x: -x[1])

            rec = {
                "w_ai": round(w_ai_val, 2),
                "rank_1": ranked[0][0],
                "rank_2": ranked[1][0],
                "rank_3": ranked[2][0],
            }
            for name, score in zip(names, c_star):
                rec[name] = round(score, 4)
            records.append(rec)

        return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# Hàm vẽ biểu đồ
# ═══════════════════════════════════════════════════════════════

def plot_topsis_comparison(result: ReadinessResult, save: bool = True) -> plt.Figure:
    """Biểu đồ thanh đôi so sánh điểm TOPSIS chuyên gia vs Entropy.

    Args:
        result: ReadinessResult từ TOPSISRanker.rank().
        save: Lưu PNG.

    Returns:
        matplotlib Figure.
    """
    names_exp = result.ranking_expert["name"].tolist()
    scores_exp = result.ranking_expert["topsis_score"].tolist()

    # Sắp xếp entropy theo thứ tự của expert
    scores_ent = [
        float(result.scores_entropy.get(n, 0)) for n in names_exp
    ]

    x = np.arange(len(names_exp))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    bars1 = ax.bar(x - w/2, scores_exp, w, label="Trọng số Chuyên gia",
                   color="#2563EB", alpha=0.85)
    bars2 = ax.bar(x + w/2, scores_ent, w, label="Trọng số Entropy",
                   color="#F59E0B", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names_exp, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Hệ số gần gũi C* (TOPSIS)")
    title_map = {"region": "6 Vùng Kinh tế-Xã hội", "sector": "10 Ngành Kinh tế"}
    ax.set_title(
        f"Xếp hạng Sẵn sàng AI/Số — {title_map.get(result.entity_type,'')}\n"
        f"Top-3 {'on dinh' if result.top3_stable else 'thay doi'} giua hai bộ trọng số",
        fontsize=12, fontweight="bold"
    )
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fname = f"m2_topsis_{result.entity_type}.png"
        fig.savefig(OUTPUTS_DIR / fname, dpi=150)
        logger.info("Đã lưu: outputs/%s", fname)
    return fig


def plot_sensitivity_heatmap(df_sens: pd.DataFrame,
                              result: ReadinessResult,
                              save: bool = True) -> plt.Figure:
    """Heatmap điểm TOPSIS khi thay đổi trọng số AI Readiness.

    Args:
        df_sens: DataFrame từ TOPSISRanker.sensitivity_ai_weight().
        result: ReadinessResult (để lấy danh sách tên).
        save: Lưu PNG.

    Returns:
        matplotlib Figure.
    """
    names = list(result.scores_expert.index)
    heat_data = df_sens[names].values.T  # (n_entities × n_w_steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Heatmap ───────────────────────────────────────────────
    sns.heatmap(
        heat_data,
        xticklabels=[f"w={v:.2f}" for v in df_sens["w_ai"]],
        yticklabels=[n[:20] for n in names],
        cmap="RdYlGn", ax=ax1, annot=True, fmt=".3f",
        linewidths=0.3, cbar_kws={"label": "C* TOPSIS"}
    )
    ax1.set_title("Điểm C* theo trọng số AI Readiness", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Trọng số w_AI")

    # ── Top-3 stability ───────────────────────────────────────
    rank_colors = {"rank_1": "#EF4444", "rank_2": "#F97316", "rank_3": "#EAB308"}
    for rank_col, color in rank_colors.items():
        y_vals = []
        for _, row in df_sens.iterrows():
            entity = row[rank_col]
            y_vals.append(names.index(entity) if entity in names else -1)
        ax2.plot(df_sens["w_ai"], y_vals, "o-",
                 color=color, lw=2, ms=8,
                 label=rank_col.replace("_", " ").title())

    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([n[:18] for n in names], fontsize=8)
    ax2.set_xlabel("Trọng số w_AI")
    ax2.set_title("Ổn định Top-3 theo w_AI", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.invert_yaxis()

    fig.suptitle("Phân tích Độ nhạy Trọng số AI Readiness — TOPSIS",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fname = f"m2_sensitivity_{result.entity_type}.png"
        fig.savefig(OUTPUTS_DIR / fname, dpi=150)
        logger.info("Đã lưu: outputs/%s", fname)
    return fig


def plot_entropy_weights(result: ReadinessResult, save: bool = True) -> plt.Figure:
    """So sánh trọng số Chuyên gia vs Entropy trên biểu đồ radar/bar.

    Args:
        result: ReadinessResult.
        save: Lưu PNG.

    Returns:
        matplotlib Figure.
    """
    labels = result.criteria_names
    w_exp = result.weights_expert
    w_ent = result.weights_entropy

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, w_exp, 0.4, label="Chuyên gia", color="#1D4ED8", alpha=0.85)
    ax.bar(x + 0.2, w_ent, 0.4, label="Entropy",    color="#D97706", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Trọng số")
    ax.set_title("So sánh Trọng số Chuyên gia vs Entropy Weight",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / f"m2_weights_{result.entity_type}.png", dpi=150)
    return fig


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def run_m2_analysis() -> Dict:
    """Chạy toàn bộ phân tích M2 — trả kết quả cho pipeline."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    from typing import Dict

    print("\n" + "="*60)
    print("  MODULE M2 — Xếp hạng Sẵn sàng AI/Số")
    print("  TOPSIS + Entropy Weight | 6 vùng & 10 ngành")
    print("="*60)

    results = {}

    for etype in ["region", "sector"]:
        ranker = TOPSISRanker(entity_type=etype)
        result = ranker.rank()
        results[etype] = result

        label = "6 Vùng KT-XH" if etype == "region" else "10 Ngành"
        print(f"\n{'─'*50}")
        print(f"  {label} — Xếp hạng theo Trọng số Chuyên gia:")
        print(result.ranking_expert[["rank","name","topsis_score"]].to_string(index=False))

        print(f"\n  {label} — Xếp hạng theo Entropy Weight:")
        print(result.ranking_entropy[["rank","name","topsis_score"]].to_string(index=False))

        stable_txt = "✅ ỔN ĐỊNH" if result.top3_stable else "⚠️  THAY ĐỔI"
        print(f"\n  Top-3: {stable_txt}")
        print(f"  Trọng số Entropy: {np.round(result.weights_entropy, 3)}")

        # Phân tích độ nhạy
        df_sens = ranker.sensitivity_ai_weight()
        if not df_sens.empty:
            print(f"\n  Độ nhạy w_AI — Top-3:")
            print(df_sens[["w_ai","rank_1","rank_2","rank_3"]].to_string(index=False))

        # Vẽ biểu đồ
        plot_topsis_comparison(result, save=True)
        plot_entropy_weights(result, save=True)
        if not df_sens.empty:
            plot_sensitivity_heatmap(df_sens, result, save=True)

    print("\n✅ Biểu đồ M2 đã lưu vào outputs/")
    return results


if __name__ == "__main__":
    run_m2_analysis()
