"""
bai12_creative_extensions.py
==============================
MỞ RỘNG SÁNG TẠO CHO TẤT CẢ 12 BÀI — AIDEOM-VN
==================================================
Mỗi hàm là một extension độc lập, đóng gói hoàn toàn,
KHÔNG chỉnh sửa code gốc. Chạy từ thư mục gốc aideom_vn/:

    python bai12_creative_extensions.py

Hoặc gọi riêng lẻ:
    from bai12_creative_extensions import bai1_creative_extension
    bai1_creative_extension()
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# BÀI 1 — Cobb-Douglas: Mô phỏng "What-If" TFP Frontier
# ══════════════════════════════════════════════════════════════════

def bai1_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 1: TFP Frontier Simulation & Growth Phase Diagram

    Ý nghĩa kinh tế: Vẽ "bản đồ pha" tăng trưởng GDP theo (TFP, Tỷ trọng số hóa D),
    cho thấy ngưỡng D* và A* cần đạt để đảm bảo mục tiêu 7%/năm đến 2030.
    Đây là công cụ hỗ trợ quyết định trực quan cho hoạch định chính sách.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 1 ---")
    print("Vẽ Phase Diagram tăng trưởng GDP theo (TFP × Số hóa D)")
    print("Mục tiêu: Xác định miền chính sách đạt 7%/năm đến 2030")

    # ── Dữ liệu gốc (copy từ bài 1) ─────────────────────────────
    years  = np.array([2020, 2021, 2022, 2023, 2024, 2025])
    Y      = np.array([8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6])
    K      = np.array([16500, 17800, 19600, 21300, 23500, 25900])
    L      = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
    D_hist = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
    AI     = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
    H      = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])

    alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07

    # TFP lịch sử
    denom = (K**alpha) * (L**beta) * (D_hist**gamma) * (AI**delta) * (H**theta)
    A_hist = Y / denom
    A_mean = A_hist.mean()

    # ── Mô phỏng What-If: lưới A × D ─────────────────────────────
    A_grid = np.linspace(A_mean * 0.7, A_mean * 1.5, 60)
    D_grid = np.linspace(10, 45, 60)
    A_2d, D_2d = np.meshgrid(A_grid, D_grid)

    # Giả định K, L, AI, H năm 2030 theo tốc độ tăng trưởng trung bình
    horizon = 5
    K30  = K[-1]  * (1 + 0.06) ** horizon
    L30  = L[-1]  * (1 + 0.006) ** horizon
    AI30 = 100.0
    H30  = 35.0

    Y30 = A_2d * (K30**alpha) * (L30**beta) * (D_2d**gamma) * (AI30**delta) * (H30**theta)
    growth_rate = ((Y30 / Y[-1]) ** (1 / horizon) - 1) * 100

    # ── Tính điểm VN 2025 và mục tiêu 2030 ───────────────────────
    A_2025 = A_hist[-1]
    D_2025 = 19.5
    D_target = 30.0
    A_target_needed = (Y[-1] * (1.07**5)) / (
        (K30**alpha) * (L30**beta) * (D_target**gamma) * (AI30**delta) * (H30**theta)
    )

    # ── Vẽ Phase Diagram ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Bài 1 — Extension: TFP × Số hóa Phase Diagram\n"
        "Miền Chính sách Đạt Mục tiêu Tăng trưởng 7%/năm (2025→2030)",
        fontsize=13, fontweight="bold"
    )

    # Contour tốc độ tăng trưởng
    ax = axes[0]
    levels = [4, 5, 6, 7, 8, 9, 10, 12]
    cf = ax.contourf(A_grid, D_grid, growth_rate, levels=30, cmap="RdYlGn")
    cs = ax.contour(A_grid, D_grid, growth_rate, levels=levels,
                    colors="white", linewidths=0.8, alpha=0.7)
    ax.clabel(cs, fmt="%.0f%%", fontsize=8, inline=True)
    plt.colorbar(cf, ax=ax, label="Tốc độ tăng trưởng GDP (%/năm)")

    # Đường mục tiêu 7%
    ax.contour(A_grid, D_grid, growth_rate, levels=[7.0],
               colors=["#FF4444"], linewidths=2.5)
    ax.text(A_mean * 1.35, 28, "Ranh giới\n7%/năm", color="#FF4444",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # Điểm hiện tại VN 2025
    ax.scatter([A_2025], [D_2025], s=200, color="#1565C0", zorder=10,
               marker="o", label=f"VN 2025 (A={A_2025:.3f}, D={D_2025}%)")
    ax.scatter([A_target_needed], [D_target], s=200, color="#E65100", zorder=10,
               marker="*", label=f"Mục tiêu 2030 (A≥{A_target_needed:.3f}, D=30%)")

    # Mũi tên lộ trình
    ax.annotate("", xy=(A_target_needed, D_target), xytext=(A_2025, D_2025),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2))
    ax.set_xlabel("TFP (A_t)", fontsize=11)
    ax.set_ylabel("Tỷ trọng Kinh tế số D (%GDP)", fontsize=11)
    ax.set_title("Phase Diagram: Tốc độ Tăng trưởng GDP 2030", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)

    # Panel 2: Sensitivity — GDP 2030 theo TFP với 3 kịch bản D
    ax2 = axes[1]
    A_range = np.linspace(A_mean * 0.7, A_mean * 1.5, 100)
    for D_scen, color, label in [
        (19.5, "#6B7280", "S1: D=19.5% (Truyền thống)"),
        (28.0, "#3B82F6", "S2: D=28% (Số hóa nhanh)"),
        (30.0, "#F59E0B", "S5: D=30% (Cân bằng tối ưu)"),
        (35.0, "#8B5CF6", "S3: D=35% (AI dẫn dắt)"),
    ]:
        Y30_line = A_range * (K30**alpha) * (L30**beta) * \
                   (D_scen**gamma) * (AI30**delta) * (H30**theta)
        g_line = ((Y30_line / Y[-1]) ** (1/5) - 1) * 100
        ax2.plot(A_range, g_line, color=color, lw=2.2, label=label)

    ax2.axhline(7.0, color="red", ls="--", lw=2, label="Mục tiêu 7%/năm")
    ax2.axvline(A_2025, color="gray", ls=":", lw=1.5, label=f"A_2025={A_2025:.3f}")
    ax2.fill_between(A_range, 7.0, 12,
                     where=(A_range >= A_mean * 0.95), alpha=0.08, color="green",
                     label="Vùng đạt mục tiêu")
    ax2.set_xlabel("TFP (A_t)", fontsize=11)
    ax2.set_ylabel("Tăng trưởng GDP TB/năm (%)", fontsize=11)
    ax2.set_title("GDP Growth 2030 theo TFP × Kịch bản Số hóa D", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(2, 13)

    plt.tight_layout()
    out = OUTPUT_DIR / "bai01_extension_phase_diagram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  → TFP cần đạt tối thiểu để đạt 7%/năm với D=30%: {A_target_needed:.4f}")
    print(f"  → TFP hiện tại 2025: {A_2025:.4f} — cần cải thiện {(A_target_needed/A_2025-1)*100:.1f}%")
    print(f"  → Biểu đồ đã lưu: {out}")


# ══════════════════════════════════════════════════════════════════
# BÀI 2 — LP Budget: Pareto Frontier GDP-Gain × Nhân lực
# ══════════════════════════════════════════════════════════════════

def bai2_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 2: Biên Pareto 2 Mục tiêu (GDP gain × Nhân lực số)

    Ý nghĩa kinh tế: Minh hoạ đánh đổi (trade-off) giữa tối đa hoá GDP gain
    và đảm bảo đầu tư nhân lực số. Đường Pareto Frontier cho thấy "chi phí
    cơ hội" thực sự của chính sách ưu tiên nhân lực — thông tin quan trọng
    cho đàm phán chính sách đa bên.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 2 ---")
    print("Vẽ Pareto Frontier: GDP gain ↔ Đầu tư Nhân lực số x3")

    try:
        import pulp
    except ImportError:
        print("  ⚠ pulp chưa cài, bỏ qua")
        return

    coeffs  = [0.85, 1.20, 0.95, 1.35]  # beta
    labels  = ["x1-Hạ tầng", "x2-AI", "x3-Nhân lực", "x4-R&D"]
    budget  = 100

    def solve_with_x3_weight(w3_extra):
        """Giải LP với penalty/bonus trên x3 để tạo Pareto front."""
        m = pulp.LpProblem("Bai2_Pareto", pulp.LpMaximize)
        x1 = pulp.LpVariable("x1", lowBound=25)
        x2 = pulp.LpVariable("x2", lowBound=15)
        x3 = pulp.LpVariable("x3", lowBound=20)
        x4 = pulp.LpVariable("x4", lowBound=10)
        # Hàm mục tiêu: GDP gain + trọng số bổ sung cho x3
        m += 0.85*x1 + 1.20*x2 + (0.95 + w3_extra)*x3 + 1.35*x4
        m += x1 + x2 + x3 + x4 <= budget
        m += x2 + x4 >= 0.35*(x1+x2+x3+x4)
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[m.status] == "Optimal":
            x3_val = pulp.value(x3)
            # GDP gain thực (không tính w3_extra)
            gdp = 0.85*pulp.value(x1) + 1.20*pulp.value(x2) + \
                  0.95*pulp.value(x3) + 1.35*pulp.value(x4)
            return gdp, x3_val
        return None, None

    # Quét w3_extra để tạo frontier
    w3_range  = np.linspace(-0.5, 1.5, 80)
    frontier  = [solve_with_x3_weight(w) for w in w3_range]
    gdp_vals  = [f[0] for f in frontier if f[0]]
    x3_vals   = [f[1] for f in frontier if f[1]]

    # Lọc Pareto-dominant (không bị dominated)
    pareto_gdp, pareto_x3 = [], []
    for g, x3v in sorted(zip(gdp_vals, x3_vals), key=lambda t: t[1]):
        if not pareto_gdp or g > pareto_gdp[-1] - 0.001:
            pareto_gdp.append(g)
            pareto_x3.append(x3v)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Bài 2 — Extension: Pareto Frontier GDP gain ↔ Nhân lực số",
                 fontsize=13, fontweight="bold")

    # Panel 1: Pareto Frontier
    ax = axes[0]
    ax.scatter(x3_vals, gdp_vals, s=30, alpha=0.4, color="#B0BEC5", label="Tất cả nghiệm LP")
    ax.plot(pareto_x3, pareto_gdp, "o-", color="#1565C0", lw=2.5,
            ms=6, label="Pareto Frontier", zorder=5)

    # Highlight điểm gốc (x3_min=20) và x3_min=30
    ax.scatter([20], [solve_with_x3_weight(-0.45)[0]], s=150, color="#E65100",
               zorder=10, marker="D", label="x3=20 (gốc)")
    ax.scatter([30], [solve_with_x3_weight(0.3)[0]], s=150, color="#2E7D32",
               zorder=10, marker="^", label="x3=30 (chính sách NL)")

    # Tô vùng "Vùng chính sách tốt"
    ax.axvspan(25, 35, alpha=0.08, color="green", label="Vùng chính sách khuyến nghị")
    ax.set_xlabel("x3 — Đầu tư Nhân lực số (nghìn tỷ VND)", fontsize=11)
    ax.set_ylabel("Z* — GDP gain (nghìn tỷ VND)", fontsize=11)
    ax.set_title("Đường Biên Pareto: GDP gain ↔ Nhân lực số", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Chi phí cơ hội biên ∂GDP/∂x3
    ax2 = axes[1]
    if len(pareto_x3) > 2:
        marginal_cost = -np.gradient(pareto_gdp, pareto_x3)
        ax2.plot(pareto_x3[1:-1], marginal_cost[1:-1],
                 "o-", color="#C62828", lw=2.5, ms=6)
        ax2.axhline(0, color="gray", ls="--", lw=1)
        ax2.fill_between(pareto_x3[1:-1], marginal_cost[1:-1], 0,
                         where=(np.array(marginal_cost[1:-1]) > 0),
                         alpha=0.15, color="red", label="Chi phí GDP khi tăng x3")
        ax2.set_xlabel("x3 — Đầu tư Nhân lực số (nghìn tỷ VND)", fontsize=11)
        ax2.set_ylabel("Chi phí cơ hội biên ∂GDP/∂x3", fontsize=11)
        ax2.set_title("Chi phí Cơ hội Biên của Đầu tư Nhân lực số", fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "bai02_extension_pareto_frontier.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  → Số điểm Pareto: {len(pareto_gdp)}")
    if pareto_gdp:
        print(f"  → GDP max (x3 thấp nhất): {max(pareto_gdp):.4f} | "
              f"GDP min (x3 cao nhất): {min(pareto_gdp):.4f}")
        print(f"  → Đánh đổi: tăng x3 thêm 10 nghìn tỷ → GDP giảm ~"
              f"{(max(pareto_gdp)-min(pareto_gdp)):.2f} nghìn tỷ")
    print(f"  → Biểu đồ đã lưu: {out}")


# ══════════════════════════════════════════════════════════════════
# BÀI 3 — Priority Index: Cluster Analysis & Radar Chart
# ══════════════════════════════════════════════════════════════════

def bai3_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 3: K-Means Clustering + Radar Chart 10 Ngành

    Ý nghĩa kinh tế: Nhóm 10 ngành thành các cụm chiến lược (đầu tàu, bắt kịp,
    cần hỗ trợ) qua K-Means, kết hợp Radar Chart đa chiều — hỗ trợ Bộ KH&ĐT
    thiết kế gói chính sách "trúng đích" theo đặc điểm nhóm ngành.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 3 ---")
    print("K-Means Clustering ngành + Radar Chart đa chiều")

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans

    sectors = ["Nông-LT-TS", "CN chế biến", "Xây dựng", "Khai khoáng",
               "Bán buôn-lẻ", "Tài chính-NH", "Logistics", "CNTT-TT",
               "Giáo dục", "Y tế"]

    raw = np.array([
        [ 3.27,  103.4, 0.35,  40.5, 13.20, 15, 18],
        [ 9.64,  241.2, 0.78, 290.9, 11.50, 55, 42],
        [ 7.45,  168.8, 0.42,   2.5,  4.80, 20, 25],
        [-1.20, 1290.5, 0.30,   8.2,  0.30, 30, 55],
        [ 7.10,  145.3, 0.55,   5.5,  7.80, 48, 38],
        [ 7.36, 1072.4, 0.85,   1.2,  0.55, 72, 52],
        [ 9.93,  321.4, 0.72,   3.1,  1.95, 42, 35],
        [ 7.85,  713.8, 0.92, 178.0,  0.62, 88, 28],
        [ 6.42,  205.7, 0.65,   0.0,  2.15, 38, 22],
        [ 6.85,  437.1, 0.60,   0.0,  0.75, 45, 18],
    ])

    # Chuẩn hoá
    scaler   = MinMaxScaler()
    raw_norm = scaler.fit_transform(raw)
    # Đảo cột Risk (cột 6)
    raw_norm[:, 6] = 1 - raw_norm[:, 6]

    # K-Means k=3
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_km = km.fit_predict(raw_norm)

    cluster_names = {0: "Đầu tàu số", 1: "Chuyển đổi sớm", 2: "Cần hỗ trợ"}
    cluster_colors = {0: "#1565C0", 1: "#F59E0B", 2: "#E53935"}

    # Radar Chart — 5 chiều đại diện
    radar_dims  = ["Tăng trưởng", "Năng suất", "Lan toả", "AI Sẵn sàng", "An toàn Rủi ro"]
    radar_idx   = [0, 1, 2, 5, 6]  # chỉ số trong raw_norm
    N_dims = len(radar_dims)
    angles = np.linspace(0, 2*np.pi, N_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("Bài 3 — Extension: Phân cụm Chiến lược 10 Ngành (K-Means k=3)\n"
                 "Kết hợp Radar Chart Đặc trưng Cụm", fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Panel 1: Scatter PCA 2D
    ax1 = fig.add_subplot(gs[0])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(raw_norm)

    for k, cname in cluster_names.items():
        idx = np.where(labels_km == k)[0]
        ax1.scatter(X_pca[idx, 0], X_pca[idx, 1], s=120,
                    color=cluster_colors[k], label=f"Cụm {k}: {cname}",
                    edgecolors="white", lw=1.5, zorder=5)
    for i, s in enumerate(sectors):
        ax1.annotate(s[:8], (X_pca[i, 0], X_pca[i, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7.5)
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    ax1.set_title("PCA 2D — Phân cụm K-Means 3 Nhóm Ngành", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # Panel 2: Radar Chart trung bình cụm
    ax2 = fig.add_subplot(gs[1], polar=True)
    for k, cname in cluster_names.items():
        idx = np.where(labels_km == k)[0]
        vals = raw_norm[idx][:, radar_idx].mean(axis=0).tolist()
        vals += vals[:1]
        ax2.plot(angles, vals, color=cluster_colors[k], lw=2.5, label=f"Cụm {k}: {cname}")
        ax2.fill(angles, vals, color=cluster_colors[k], alpha=0.12)

    ax2.set_thetagrids(np.degrees(angles[:-1]), radar_dims, fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_title("Radar: Profile Trung bình Cụm", fontweight="bold", pad=15)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax2.grid(alpha=0.4)

    plt.savefig(OUTPUT_DIR / "bai03_extension_cluster_radar.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  Phân cụm K-Means (k=3):")
    for k, cname in cluster_names.items():
        members = [sectors[i] for i in np.where(labels_km == k)[0]]
        print(f"    Cụm {k} — {cname}: {members}")
    print(f"  → Gợi ý chính sách: thiết kế 3 gói hỗ trợ CĐS riêng biệt cho từng cụm")
    print(f"  → Biểu đồ đã lưu: outputs/bai03_extension_cluster_radar.png")


# ══════════════════════════════════════════════════════════════════
# BÀI 4 — LP Region: Gradient Sensitivity + Equity-Efficiency Curve
# ══════════════════════════════════════════════════════════════════

def bai4_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 4: Equity-Efficiency Frontier (λ-sweep)

    Ý nghĩa kinh tế: Quét hệ số λ (equity constraint) từ 0 → 0.9 để vẽ
    đường biên Hiệu quả-Công bằng. Cung cấp cho nhà hoạch định chính sách
    thông tin định lượng: "Chấp nhận giảm X% GDP gain để đổi lấy Y% cải
    thiện bất bình đẳng vùng" — ngôn ngữ kinh tế chính trị thực tế.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 4 ---")
    print("Quét λ (equity): Vẽ Equity-Efficiency Frontier")

    try:
        import pulp
    except ImportError:
        print("  ⚠ pulp chưa cài, bỏ qua")
        return

    regions = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
    items   = ["I", "D", "AI", "H"]
    beta    = np.array([
        [1.15, 0.85, 0.55, 1.30],
        [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15],
        [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00],
        [1.10, 0.85, 0.65, 1.25],
    ])
    D0 = np.array([38, 78, 55, 32, 82, 48])
    gamma_eq = 0.002

    def solve_lambda(lam):
        m = pulp.LpProblem(f"Eq_{lam:.2f}", pulp.LpMaximize)
        x = {(r, j): pulp.LpVariable(f"x_{r}_{j}", lowBound=0)
             for r in regions for j in items}
        ri = {r: i for i, r in enumerate(regions)}
        ji = {j: i for i, j in enumerate(items)}
        m += pulp.lpSum(beta[ri[r], ji[j]] * x[(r, j)] for r in regions for j in items)
        m += pulp.lpSum(x[(r, j)] for r in regions for j in items) <= 50000
        for r in regions:
            m += pulp.lpSum(x[(r, j)] for j in items) >= 5000
            m += pulp.lpSum(x[(r, j)] for j in items) <= 12000
        m += pulp.lpSum(x[(r, "H")] for r in regions) >= 12000

        if lam > 0:
            Dmax = pulp.LpVariable("Dmax", lowBound=0)
            for r in regions:
                m += D0[ri[r]] + gamma_eq * x[(r, "D")] <= Dmax
                m += D0[ri[r]] + gamma_eq * x[(r, "D")] >= lam * Dmax

        m.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[m.status] != "Optimal":
            return None, None

        Z = pulp.value(m.objective)
        # Tính Gini xấp xỉ từ phân bổ vùng
        allocs = [pulp.value(pulp.lpSum(x[(r, j)] for j in items)) for r in regions]
        allocs = np.array([a if a else 0 for a in allocs])
        n = len(allocs)
        gini = (np.sum(np.abs(np.subtract.outer(allocs, allocs))) /
                (2 * n * allocs.sum() + 1e-9))
        return Z, gini

    lambdas   = np.arange(0, 0.85, 0.05)
    z_vals, gini_vals = [], []
    for lam in lambdas:
        Z, gini = solve_lambda(lam)
        if Z:
            z_vals.append(Z)
            gini_vals.append(gini)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Bài 4 — Extension: Equity-Efficiency Frontier\n"
                 "Đánh đổi GDP gain ↔ Bất bình đẳng vùng (λ-sweep)",
                 fontsize=13, fontweight="bold")

    # Panel 1: Scatter Gini vs GDP gain
    ax = axes[0]
    sc = ax.scatter(gini_vals, z_vals, c=lambdas[:len(z_vals)],
                    cmap="RdYlGn_r", s=80, zorder=5)
    plt.colorbar(sc, ax=ax, label="λ (equity constraint strength)")
    ax.plot(gini_vals, z_vals, "--", color="gray", alpha=0.6, lw=1.5)
    # Highlight điểm λ=0 (chỉ hiệu quả) và λ=0.6
    if len(z_vals) >= 2:
        ax.scatter([gini_vals[0]], [z_vals[0]], s=200, color="#C62828",
                   marker="D", zorder=10, label="λ=0 (thuần hiệu quả)")
        idx55 = min(range(len(lambdas[:len(z_vals)])),
                    key=lambda i: abs(lambdas[i] - 0.55))
        ax.scatter([gini_vals[idx55]], [z_vals[idx55]], s=200, color="#1565C0",
                   marker="*", zorder=10, label=f"λ=0.55 (cân bằng)")
    ax.set_xlabel("Gini Index (Bất bình đẳng phân bổ vùng)", fontsize=11)
    ax.set_ylabel("Z* GDP gain (tỷ VND)", fontsize=11)
    ax.set_title("Đường biên Equity-Efficiency", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.invert_xaxis()

    # Panel 2: Z*(λ) và Gini(λ) dual-axis
    ax2 = axes[1]
    lam_plot = lambdas[:len(z_vals)]
    color1, color2 = "#1565C0", "#E65100"
    l1, = ax2.plot(lam_plot, z_vals, "o-", color=color1, lw=2.5, ms=7, label="Z* GDP gain")
    ax2.set_ylabel("Z* GDP gain (tỷ VND)", color=color1, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color1)
    ax3 = ax2.twinx()
    l2, = ax3.plot(lam_plot, gini_vals, "s--", color=color2, lw=2.5, ms=7, label="Gini Index")
    ax3.set_ylabel("Gini Index (bất bình đẳng)", color=color2, fontsize=11)
    ax3.tick_params(axis="y", labelcolor=color2)
    ax2.set_xlabel("λ (hệ số công bằng vùng)", fontsize=11)
    ax2.set_title("Z* và Gini theo λ (dual-axis)", fontweight="bold")
    lines = [l1, l2]
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="center left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bai04_extension_equity_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close()

    if z_vals:
        print(f"  → Z* tối đa (λ=0): {z_vals[0]:,.1f} tỷ VND | Gini={gini_vals[0]:.4f}")
        print(f"  → Z* với λ=0.55: {z_vals[min(10,len(z_vals)-1)]:,.1f} tỷ VND | "
              f"Gini={gini_vals[min(10,len(z_vals)-1)]:.4f}")
    print(f"  → Biểu đồ đã lưu: outputs/bai04_extension_equity_efficiency.png")


# ══════════════════════════════════════════════════════════════════
# BÀI 5 — MIP Projects: Monte Carlo Risk-Adjusted NPV
# ══════════════════════════════════════════════════════════════════

def bai5_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 5: Monte Carlo Risk-Adjusted NPV Portfolio

    Ý nghĩa kinh tế: Mô phỏng 5000 lần với NPV ngẫu nhiên (lognormal) cho
    từng dự án để tính phân phối Z* danh mục. Kết quả cho thấy VaR (Value
    at Risk) và CVaR của danh mục được chọn — thông tin quan trọng cho Bộ
    Tài chính trong quản lý rủi ro ngân sách công nghệ.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 5 ---")
    print("Monte Carlo 5,000 lần → Phân phối Z* danh mục dự án tối ưu")

    P = list(range(1, 16))
    B_base  = {1:21500, 2:20800, 3:32500, 4:9200, 5:6800,
               6:11400, 7:12200, 8:28500, 9:5800, 10:13800,
               11:8500, 12:16200, 13:35000, 14:7500, 15:3800}
    C       = {1:12000, 2:11500, 3:18000, 4:4500, 5:3200,
               6:5800, 7:6500, 8:15000, 9:2500, 10:7200,
               11:4800, 12:8500, 13:20000, 14:3800, 15:1500}
    prob    = {1:0.85, 2:0.85, 3:0.85, 4:0.75, 5:0.75,
               6:0.80, 7:0.80, 8:0.65, 9:0.80, 10:0.80,
               11:0.80, 12:0.80, 13:0.65, 14:0.80, 15:0.80}
    # Danh mục được chọn ở bài gốc (giả định kết quả điển hình)
    selected_base = [1, 3, 6, 7, 8, 12, 14]
    selected_alt  = [3, 8, 12, 13, 14, 6, 7]  # nếu P13 được chọn

    np.random.seed(42)
    N_SIM = 5000

    def simulate_portfolio(selected, n_sim=N_SIM):
        """Mô phỏng Z* với NPV ngẫu nhiên (lognormal, sigma=20%)."""
        z_sims = []
        for _ in range(n_sim):
            z = 0
            for i in selected:
                # Lognormal: mean=B_base[i], sigma=20%
                b_rand = np.random.lognormal(
                    mean=np.log(B_base[i]) - 0.02,
                    sigma=0.20
                )
                # Xác suất hoàn thành
                if np.random.random() < prob[i]:
                    z += b_rand
            z_sims.append(z)
        return np.array(z_sims)

    z_base = simulate_portfolio(selected_base)
    z_alt  = simulate_portfolio(selected_alt)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Bài 5 — Extension: Monte Carlo NPV Portfolio Analysis\n"
                 "Phân phối Z* Danh mục dự án CĐS Quốc gia (N=5,000 mô phỏng)",
                 fontsize=13, fontweight="bold")

    # Panel 1: Histogram phân phối Z* - 2 danh mục
    ax = axes[0]
    ax.hist(z_base/1000, bins=60, alpha=0.65, color="#1565C0",
            label=f"Danh mục gốc (n={len(selected_base)})", density=True)
    ax.hist(z_alt/1000, bins=60, alpha=0.65, color="#E65100",
            label=f"Danh mục thay thế (n={len(selected_alt)})", density=True)
    # VaR 95%
    var95_base = np.percentile(z_base, 5)/1000
    var95_alt  = np.percentile(z_alt, 5)/1000
    ax.axvline(var95_base, color="#1565C0", ls="--", lw=2,
               label=f"VaR 95% gốc: {var95_base:.0f}k tỷ")
    ax.axvline(var95_alt, color="#E65100", ls="--", lw=2,
               label=f"VaR 95% alt: {var95_alt:.0f}k tỷ")
    ax.set_xlabel("Z* NPV danh mục (nghìn tỷ VND)", fontsize=10)
    ax.set_ylabel("Mật độ xác suất", fontsize=10)
    ax.set_title("Phân phối Z* Monte Carlo", fontweight="bold")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # Panel 2: CDF so sánh
    ax2 = axes[1]
    z_sorted_b = np.sort(z_base)/1000
    z_sorted_a = np.sort(z_alt)/1000
    cdf = np.arange(1, N_SIM+1) / N_SIM
    ax2.plot(z_sorted_b, cdf, color="#1565C0", lw=2.5, label="Danh mục gốc")
    ax2.plot(z_sorted_a, cdf, color="#E65100", lw=2.5, label="Danh mục alt (P13)")
    ax2.axhline(0.05, color="gray", ls=":", lw=1.5, label="Ngưỡng 5% (VaR)")
    ax2.axhline(0.50, color="gray", ls="-.", lw=1, label="Trung vị (P50)")
    ax2.set_xlabel("Z* NPV (nghìn tỷ VND)", fontsize=10)
    ax2.set_ylabel("CDF F(Z)", fontsize=10)
    ax2.set_title("CDF — Stochastic Dominance", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Panel 3: Risk metrics bảng
    ax3 = axes[2]
    ax3.axis("off")
    metrics_data = []
    for name, z_arr in [("Danh mục gốc", z_base), ("Danh mục alt", z_alt)]:
        p5, p25, p50, p75, p95 = np.percentile(z_arr, [5,25,50,75,95])
        var95 = p5
        cvar95 = z_arr[z_arr <= var95].mean() if (z_arr <= var95).any() else var95
        metrics_data.append([
            name,
            f"{z_arr.mean()/1000:,.1f}",
            f"{z_arr.std()/1000:,.1f}",
            f"{p50/1000:,.1f}",
            f"{var95/1000:,.1f}",
            f"{cvar95/1000:,.1f}",
        ])
    cols = ["Danh mục", "Mean", "Std", "P50", "VaR 95%", "CVaR 95%"]
    tbl = ax3.table(cellText=metrics_data, colLabels=cols,
                    cellLoc="center", loc="center",
                    bbox=[0, 0.3, 1, 0.5])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1565C0"); cell.set_text_props(color="white")
        elif row % 2 == 0:
            cell.set_facecolor("#EFF6FF")
    ax3.set_title("Risk Metrics (nghìn tỷ VND)", fontweight="bold", y=0.85)
    note = (f"* Lognormal simulation σ=20%\n"
            f"  Tỷ lệ hoàn thành đúng tiến độ theo từng dự án\n"
            f"  Đơn vị: nghìn tỷ VND")
    ax3.text(0.5, 0.15, note, transform=ax3.transAxes, ha="center",
             fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bai05_extension_mc_npv.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  → Danh mục gốc — Mean: {z_base.mean()/1000:,.1f}k tỷ | VaR95: {np.percentile(z_base,5)/1000:,.1f}k tỷ")
    print(f"  → Danh mục alt  — Mean: {z_alt.mean()/1000:,.1f}k tỷ  | VaR95: {np.percentile(z_alt,5)/1000:,.1f}k tỷ")
    print(f"  → Biểu đồ đã lưu: outputs/bai05_extension_mc_npv.png")


# ══════════════════════════════════════════════════════════════════
# BÀI 6 — TOPSIS: Bootstrap Confidence Interval cho Ranking
# ══════════════════════════════════════════════════════════════════

def bai6_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 6: Bootstrap Confidence Interval cho TOPSIS Ranking

    Ý nghĩa kinh tế: Xếp hạng TOPSIS thường bị coi là "tuyệt đối" nhưng thực ra
    phụ thuộc nhiễu đo lường. Bootstrap 2,000 lần cho thấy khoảng tin cậy 95%
    của C* từng vùng — vùng nào xếp hạng ổn định, vùng nào cần thêm dữ liệu.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 6 ---")
    print("Bootstrap 2,000 lần → CI 95% cho TOPSIS C* ranking")

    regions = ["T.du MN Bắc", "ĐB sông Hồng", "Bắc TBộ+DH",
               "Tây Nguyên", "Đông Nam Bộ", "ĐB Cửu Long"]
    raw = np.array([
        [57.0,   3.5, 38, 22, 21.5, 0.18, 72, 0.405],
        [152.3, 20.0, 78, 68, 36.8, 0.85, 92, 0.358],
        [ 87.5,  8.2, 55, 40, 27.5, 0.32, 84, 0.372],
        [ 68.9,  0.8, 32, 18, 18.2, 0.15, 68, 0.412],
        [158.9, 18.5, 82, 75, 42.5, 0.78, 94, 0.385],
        [ 80.5,  2.1, 48, 30, 16.8, 0.22, 78, 0.392],
    ])
    is_benefit = [True, True, True, True, True, True, True, False]
    w_expert   = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

    def topsis(X, w, ben):
        norms = np.sqrt((X**2).sum(axis=0))
        R = X / (norms + 1e-12)
        V = R * w
        A_star = np.where(ben, V.max(0), V.min(0))
        A_neg  = np.where(ben, V.min(0), V.max(0))
        S_plus  = np.sqrt(((V - A_star)**2).sum(1))
        S_minus = np.sqrt(((V - A_neg )**2).sum(1))
        return S_minus / (S_plus + S_minus + 1e-12)

    # Bootstrap: nhiễu đo lường ±5% trên dữ liệu
    np.random.seed(42)
    N_BOOT = 2000
    c_boot = np.zeros((N_BOOT, 6))
    for b in range(N_BOOT):
        noise = np.random.normal(1.0, 0.05, raw.shape)
        X_b   = raw * noise
        X_b   = np.abs(X_b)  # đảm bảo dương
        c_boot[b] = topsis(X_b, w_expert, is_benefit)

    c_mean = c_boot.mean(0)
    c_lo   = np.percentile(c_boot, 2.5, axis=0)
    c_hi   = np.percentile(c_boot, 97.5, axis=0)
    rank_boot = (6 - c_boot.argsort(axis=1).argsort(axis=1))
    rank_mean = rank_boot.mean(0)
    rank_std  = rank_boot.std(0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Bài 6 — Extension: Bootstrap CI 95% cho TOPSIS C* Ranking\n"
                 "Đo lường Độ ổn định Xếp hạng 6 Vùng (N=2,000 mẫu)",
                 fontsize=13, fontweight="bold")

    # Panel 1: C* với CI
    ax = axes[0]
    sort_idx = np.argsort(c_mean)[::-1]
    x_pos    = np.arange(6)
    colors_ci = ["#1565C0" if i in [0,1,2] else "#78909C" for i in range(6)]
    bars = ax.bar(x_pos, c_mean[sort_idx], color=[colors_ci[i] for i in sort_idx],
                  alpha=0.85, edgecolor="white")
    ax.errorbar(x_pos, c_mean[sort_idx],
                yerr=[c_mean[sort_idx]-c_lo[sort_idx], c_hi[sort_idx]-c_mean[sort_idx]],
                fmt="none", color="#C62828", capsize=6, lw=2, label="CI 95%")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([regions[i][:12] for i in sort_idx], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("C* (Bootstrap Mean ± CI 95%)", fontsize=10)
    ax.set_title("TOPSIS C* với Khoảng Tin cậy 95% (Bootstrap)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # Panel 2: Box plot phân phối rank
    ax2 = axes[1]
    sort_by_rank = np.argsort(rank_mean)
    bp_data = [rank_boot[:, i] for i in sort_by_rank]
    bp = ax2.boxplot(bp_data, labels=[regions[i][:12] for i in sort_by_rank],
                     patch_artist=True, notch=True,
                     medianprops=dict(color="#C62828", lw=2))
    palette = ["#1565C0", "#3B82F6", "#93C5FD", "#B0BEC5", "#90A4AE", "#78909C"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax2.set_xticklabels([regions[i][:12] for i in sort_by_rank], rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Hạng (1 = tốt nhất)", fontsize=10)
    ax2.set_title("Phân phối Hạng Bootstrap (Boxplot có notch)", fontweight="bold")
    ax2.invert_yaxis(); ax2.grid(alpha=0.3, axis="y")
    ax2.text(0.02, 0.98, "Hộp hẹp = Xếp hạng ổn định\nHộp rộng = Cần thêm dữ liệu",
             transform=ax2.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bai06_extension_bootstrap_ci.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  Kết quả Bootstrap (N=2,000):")
    print(f"  {'Vùng':<18} {'C* Mean':>10} {'CI 2.5%':>10} {'CI 97.5%':>10} {'Rank Mean':>11} {'Rank Std':>10}")
    print("  " + "-"*63)
    for i in np.argsort(rank_mean):
        print(f"  {regions[i]:<18} {c_mean[i]:>10.4f} {c_lo[i]:>10.4f} "
              f"{c_hi[i]:>10.4f} {rank_mean[i]:>11.2f} {rank_std[i]:>10.2f}")
    print(f"\n  → Đông Nam Bộ & ĐB sông Hồng: rank ổn định (std thấp)")
    print(f"  → Biểu đồ đã lưu: outputs/bai06_extension_bootstrap_ci.png")


# ══════════════════════════════════════════════════════════════════
# BÀI 7 — NSGA-II: Hypervolume Indicator + Knee Point Detection
# ══════════════════════════════════════════════════════════════════

def bai7_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 7: Hypervolume Indicator + Knee Point Detection

    Ý nghĩa kinh tế: Hypervolume đo "diện tích Pareto Front" — chỉ số chất
    lượng tối ưu đa mục tiêu được dùng phổ biến trong học thuật. Knee Point
    tự động tìm nghiệm "thoả hiệp tự nhiên nhất" không cần đặt trọng số —
    hữu ích khi nhà hoạch định chính sách không muốn/không thể chỉ định w.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 7 ---")
    print("Sinh Pareto Front mô phỏng + Hypervolume + Knee Point Detection")

    # Sinh Pareto Front 2D mô phỏng (f1=GDP, f2=Gini) thay vì chạy NSGA-II thật
    np.random.seed(42)
    n_pts = 120
    t     = np.linspace(0, np.pi/2, n_pts)
    f1    = 60000 + 8000 * np.cos(t) + np.random.normal(0, 300, n_pts)
    f2    = 500  + 3000 * np.sin(t)  + np.random.normal(0, 80,  n_pts)

    # Pareto filter (non-dominated trong tối đa f1, tối thiểu f2)
    pareto_mask = np.ones(n_pts, dtype=bool)
    for i in range(n_pts):
        for j in range(n_pts):
            if i != j and f1[j] >= f1[i] and f2[j] <= f2[i] and \
               (f1[j] > f1[i] or f2[j] < f2[i]):
                pareto_mask[i] = False
                break
    pf1 = f1[pareto_mask]
    pf2 = f2[pareto_mask]
    sort_idx = np.argsort(pf1)
    pf1, pf2 = pf1[sort_idx], pf2[sort_idx]

    # Hypervolume (reference point = worst)
    ref_f1, ref_f2 = pf1.min() * 0.95, pf2.max() * 1.05
    hv = 0.0
    prev_f1 = ref_f1
    for i in range(len(pf1)):
        hv += (pf1[i] - prev_f1) * (ref_f2 - pf2[i])
        prev_f1 = pf1[i]

    # Knee Point: điểm có khoảng cách Euclid lớn nhất đến đường nối 2 đầu
    p_start = np.array([pf1[0],  pf2[0]])
    p_end   = np.array([pf1[-1], pf2[-1]])
    line_vec    = p_end - p_start
    line_len    = np.linalg.norm(line_vec)
    pts         = np.column_stack([pf1, pf2])
    # Chuẩn hoá về [0,1] trước khi tính khoảng cách
    pts_n = (pts - pts.min(0)) / (pts.max(0) - pts.min(0) + 1e-9)
    ps_n  = (p_start - pts.min(0)) / (pts.max(0) - pts.min(0) + 1e-9)
    pe_n  = (p_end   - pts.min(0)) / (pts.max(0) - pts.min(0) + 1e-9)
    lv_n  = pe_n - ps_n
    ll_n  = np.linalg.norm(lv_n)
    dists = np.abs(np.cross(pts_n - ps_n, lv_n)) / (ll_n + 1e-9)
    knee_idx  = np.argmax(dists)
    knee_f1   = pf1[knee_idx]
    knee_f2   = pf2[knee_idx]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Bài 7 — Extension: Hypervolume Indicator + Knee Point Detection\n"
                 "Chất lượng Pareto Front & Nghiệm Thoả hiệp Tự nhiên",
                 fontsize=13, fontweight="bold")

    # Panel 1: Pareto Front với Hypervolume shading
    ax = axes[0]
    ax.fill_between([ref_f1] + list(pf1) + [pf1[-1]],
                    [ref_f2] + list(pf2) + [ref_f2],
                    ref_f2, alpha=0.15, color="#1565C0", label=f"Hypervolume = {hv:,.0f}")
    ax.plot(pf1, pf2, "o-", color="#1565C0", lw=2, ms=5, label="Pareto Front")
    ax.scatter([knee_f1], [knee_f2], s=250, color="#C62828", marker="*",
               zorder=10, label=f"Knee Point\nGDP={knee_f1:,.0f} | Gini={knee_f2:.0f}")
    ax.scatter([pf1[0]], [pf2[0]], s=120, color="#E65100", marker="D",
               zorder=9, label="Max GDP gain")
    ax.scatter([pf1[-1]], [pf2[-1]], s=120, color="#2E7D32", marker="^",
               zorder=9, label="Min Gini MAD")
    # Khoảng cách từ knee đến đường nối 2 đầu
    ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]],
            "--", color="gray", lw=1.5, alpha=0.6, label="Đường nối 2 cực")
    ax.annotate("", xy=(knee_f1, knee_f2),
                xytext=(p_start[0] + (p_end[0]-p_start[0])*(knee_idx/len(pf1)),
                        p_start[1] + (p_end[1]-p_start[1])*(knee_idx/len(pf1))),
                arrowprops=dict(arrowstyle="<->", color="#C62828", lw=1.5))
    ax.set_xlabel("f1: GDP gain (tỷ VND) — Tối đa hóa →", fontsize=10)
    ax.set_ylabel("f2: Gini MAD (bất bình đẳng) — Tối thiểu hóa →", fontsize=10)
    ax.set_title("Pareto Front 2D với Hypervolume & Knee Point", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3)

    # Panel 2: Curvature dọc Pareto Front
    ax2 = axes[1]
    if len(pf1) > 3:
        # Tính độ cong xấp xỉ = thay đổi góc
        angles   = np.arctan2(np.diff(pf2), np.diff(pf1))
        curv     = np.abs(np.diff(angles))
        x_curv   = (np.arange(len(curv)) / len(curv)) * 100
        ax2.plot(x_curv, curv, color="#6A1B9A", lw=2.5, label="Độ cong Pareto")
        ax2.fill_between(x_curv, curv, alpha=0.2, color="#6A1B9A")
        ax2.axvline(knee_idx / len(pf1) * 100, color="#C62828", ls="--", lw=2,
                    label=f"Knee Point ({knee_idx/len(pf1)*100:.0f}%)")
        ax2.set_xlabel("Vị trí dọc Pareto Front (%)", fontsize=10)
        ax2.set_ylabel("Độ cong (thay đổi góc tuyệt đối)", fontsize=10)
        ax2.set_title("Phân tích Độ cong — Xác định Knee tự động", fontweight="bold")
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
        ax2.text(0.05, 0.95,
                 f"Knee Point tại vị trí {knee_idx/len(pf1)*100:.0f}%\n"
                 f"GDP={knee_f1:,.0f} tỷ | Gini={knee_f2:.0f}\n"
                 f"Hypervolume={hv:,.0f}",
                 transform=ax2.transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bai07_extension_hypervolume_knee.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  → Hypervolume (2D GDP×Gini): {hv:,.0f}")
    print(f"  → Knee Point: GDP gain={knee_f1:,.0f} tỷ | Gini MAD={knee_f2:.1f}")
    print(f"  → Biểu đồ đã lưu: outputs/bai07_extension_hypervolume_knee.png")


# ══════════════════════════════════════════════════════════════════
# BÀI 8 — Dynamic Opt: Stochastic DP với Value Function Iteration
# ══════════════════════════════════════════════════════════════════

def bai8_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 8: Value Function Iteration (VFI) — Bellman Equation

    Ý nghĩa kinh tế: Minh họa phương pháp Value Function Iteration — thuật toán
    nền tảng của kinh tế học vĩ mô tính toán (computational macroeconomics).
    Tìm chính sách tiêu dùng tối ưu C*(K) theo vốn — minh chứng "quy tắc vàng"
    Ramsey có thể suy ra bằng số chứ không chỉ bằng giải tích.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 8 ---")
    print("Value Function Iteration (VFI) — Bellman Equation trên lưới vốn K")

    from scipy.interpolate import interp1d

    # Tham số đơn giản hoá (1 chiều: chỉ K)
    alpha_k = 0.33; delta_K = 0.05; rho = 0.97
    A0      = 30.9; L0 = 53.9; D0 = 20.3; AI0 = 86.0; H0 = 30.0
    beta_rest = 0.42; gamma_ = 0.10; delta_ = 0.08; theta_ = 0.07

    # Hàm sản xuất rút gọn: f(K) = A*K^alpha * (các hằng số)
    const_rest = A0 * (L0**beta_rest) * (D0**gamma_) * (AI0**delta_) * (H0**theta_)
    def f(K):
        return const_rest * (np.maximum(K, 1e-3) ** alpha_k)

    # Lưới vốn
    K_min, K_max = 5000, 60000
    N_K = 80
    K_grid = np.linspace(K_min, K_max, N_K)

    # VFI
    V = np.zeros(N_K)
    policy_C   = np.zeros(N_K)
    policy_K_next = np.zeros(N_K)

    MAX_ITER = 300; TOL = 1e-5
    for it in range(MAX_ITER):
        V_old  = V.copy()
        V_interp = interp1d(K_grid, V_old, kind="linear",
                            fill_value="extrapolate")
        for ki, K in enumerate(K_grid):
            Y = f(K)
            # K_next ∈ [delta*K, (1-delta)*K + Y]
            K_lo = (1 - delta_K) * K
            K_hi = min((1 - delta_K) * K + Y * 0.6, K_max)
            K_candidates = np.linspace(max(K_lo, K_min), K_hi, 50)
            C_candidates = Y + (1 - delta_K) * K - K_candidates
            # Chỉ lấy C > 0
            valid = C_candidates > 0.1
            if not valid.any():
                continue
            C_v  = C_candidates[valid]
            Kn_v = K_candidates[valid]
            vals = np.log(C_v) + rho * V_interp(Kn_v)
            best = np.argmax(vals)
            V[ki] = vals[best]
            policy_C[ki]      = C_v[best]
            policy_K_next[ki] = Kn_v[best]

        diff = np.max(np.abs(V - V_old))
        if diff < TOL:
            print(f"  VFI hội tụ sau {it+1} vòng lặp (max|ΔV|={diff:.2e})")
            break

    # Vốn steady-state: K_{t+1} = K_t → K* = f(K*) - C*(K*) + (1-δ)K*
    K_ss_idx = np.argmin(np.abs(policy_K_next - K_grid))
    K_ss = K_grid[K_ss_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Bài 8 — Extension: Value Function Iteration (VFI) — Bellman Equation\n"
                 "Tìm Chính sách Tiêu dùng Tối ưu C*(K) theo Vốn",
                 fontsize=13, fontweight="bold")

    # Panel 1: Value Function
    ax = axes[0]
    ax.plot(K_grid, V, color="#1565C0", lw=2.5, label="V*(K)")
    ax.axvline(K_ss, color="red", ls="--", lw=2, label=f"K* steady-state ≈ {K_ss:,.0f}")
    ax.set_xlabel("Vốn K (nghìn tỷ VND)", fontsize=10)
    ax.set_ylabel("V*(K) — Welfare tối đa", fontsize=10)
    ax.set_title("Value Function V*(K)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 2: Policy Function C*(K) và K'*(K)
    ax2 = axes[1]
    ax2.plot(K_grid, policy_C, color="#2E7D32", lw=2.5, label="C*(K) — tiêu dùng tối ưu")
    ax2.plot(K_grid, policy_K_next - K_grid, color="#E65100", lw=2,
             ls="--", label="K'(K) − K — đầu tư ròng")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.axvline(K_ss, color="red", ls=":", lw=1.5)
    ax2.set_xlabel("Vốn K (nghìn tỷ VND)", fontsize=10)
    ax2.set_ylabel("nghìn tỷ VND", fontsize=10)
    ax2.set_title("Policy Functions: C*(K) và Đầu tư ròng", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # Panel 3: Phase diagram K_{t+1} vs K_t
    ax3 = axes[2]
    ax3.plot(K_grid, policy_K_next, color="#6A1B9A", lw=2.5, label="K' = g*(K)")
    ax3.plot(K_grid, K_grid, color="gray", ls="--", lw=1.5, label="45° (K'=K, steady-state)")
    ax3.axvline(K_ss, color="red", ls=":", lw=1.5, label=f"K*={K_ss:,.0f}")
    # Vẽ quỹ đạo hội tụ từ K0=27500
    K_traj = [27500.0]
    for _ in range(25):
        K_next_val = float(interp1d(K_grid, policy_K_next, fill_value="extrapolate")(K_traj[-1]))
        K_traj.append(np.clip(K_next_val, K_min, K_max))
    for i in range(0, len(K_traj)-1, 2):
        ax3.annotate("", xy=(K_traj[i+1], K_traj[i+1]),
                     xytext=(K_traj[i], K_traj[i]),
                     arrowprops=dict(arrowstyle="->", color="#C62828", lw=1))
    ax3.plot(K_traj[:-1], K_traj[1:], "r.", ms=4, label="Quỹ đạo từ K0=27,500")
    ax3.set_xlabel("K_t (nghìn tỷ VND)", fontsize=10)
    ax3.set_ylabel("K_{t+1} (nghìn tỷ VND)", fontsize=10)
    ax3.set_title("Phase Diagram: Hội tụ về Steady-State K*", fontweight="bold")
    ax3.legend(fontsize=7.5); ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bai08_extension_vfi_bellman.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  → Steady-state vốn K* ≈ {K_ss:,.0f} nghìn tỷ VND")
    print(f"  → Tiêu dùng tối ưu tại K*: C*(K*) ≈ {float(interp1d(K_grid,policy_C)(K_ss)):,.1f}")
    print(f"  → Biểu đồ đã lưu: outputs/bai08_extension_vfi_bellman.png")


# ══════════════════════════════════════════════════════════════════
# BÀI 9 — Lao động: Transition Matrix Markov Chain Lao động
# ══════════════════════════════════════════════════════════════════

def bai9_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 9: Markov Chain Chuyển dịch Cơ cấu Lao động

    Ý nghĩa kinh tế: Mô hình hoá quá trình chuyển dịch lao động giữa các trạng
    thái (Có việc, Mất việc AI, Đang đào tạo lại, Việc mới AI) như một Markov
    Chain. Tính phân phối dừng (steady-state) để dự báo dài hạn tỷ lệ lao động
    trong từng trạng thái — thông tin quan trọng cho chính sách bảo hiểm thất nghiệp.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 9 ---")
    print("Markov Chain 4 trạng thái lao động → Phân phối dừng theo ngành")

    sectors = ["Nông-LT-TS", "CN chế biến", "Xây dựng",
               "Bán buôn-lẻ", "Tài chính-NH", "Logistics", "CNTT-TT", "Giáo dục"]
    risk  = np.array([0.18, 0.42, 0.25, 0.38, 0.52, 0.35, 0.28, 0.22])
    labor = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])

    # Trạng thái: 0=Có việc, 1=Mất việc/AI thay, 2=Đào tạo lại, 3=Việc mới AI
    # Ma trận chuyển tiếp P[i,j] = xác suất từ i → j, phụ thuộc risk
    def build_transition(r):
        """Xây dựng ma trận Markov 4 trạng thái theo mức risk r."""
        p_displace  = r * 0.15          # mỗi quý: r% bị thay
        p_retrain   = 0.35              # mỗi quý 35% mất việc vào đào tạo
        p_new_job   = 0.40              # đào tạo xong tìm được việc mới
        p_return    = 0.20              # việc mới AI quay lại "stable"

        P = np.array([
            [1 - p_displace,  p_displace,   0,          0       ],  # Có việc
            [0,               1-p_retrain,  p_retrain,  0       ],  # Mất việc
            [0,               0,            1-p_new_job, p_new_job], # Đào tạo
            [p_return,        0,            0,          1-p_return], # Việc mới AI
        ])
        return P

    # Tính steady-state: giải Pπ = π, Σπ = 1
    def steady_state(P):
        n = P.shape[0]
        A = (P.T - np.eye(n))
        A[-1] = 1
        b = np.zeros(n); b[-1] = 1
        pi = np.linalg.solve(A, b)
        return np.clip(pi, 0, 1)

    steady_states = []
    for r in risk:
        P = build_transition(r)
        pi = steady_state(P)
        steady_states.append(pi)
    ss = np.array(steady_states)

    # Tính số lao động theo trạng thái (triệu người)
    state_names = ["Ổn định", "Mất việc/AI", "Đào tạo lại", "Việc mới AI"]
    labor_by_state = ss * labor[:, None]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Bài 9 — Extension: Markov Chain Chuyển dịch Cơ cấu Lao động\n"
                 "Phân phối Dừng (Steady-State) 4 Trạng thái theo 8 Ngành",
                 fontsize=13, fontweight="bold")

    # Panel 1: Stacked bar steady-state distribution
    ax = axes[0]
    colors_ss = ["#2E7D32", "#C62828", "#F59E0B", "#1565C0"]
    bottom = np.zeros(8)
    x_pos  = np.arange(8)
    for j, (sname, color) in enumerate(zip(state_names, colors_ss)):
        vals = ss[:, j] * 100
        bars = ax.bar(x_pos, vals, bottom=bottom, color=color,
                      label=sname, alpha=0.85, edgecolor="white")
        bottom += vals
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s[:10] for s in sectors], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% Lao động ngành trong trạng thái", fontsize=10)
    ax.set_title("Phân phối Dừng Markov theo Ngành (%)", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3, axis="y")
    ax.axhline(100, color="gray", lw=0.5)

    # Panel 2: Heatmap số lao động theo trạng thái
    ax2 = axes[1]
    import matplotlib.colors as mcolors
    df_heat = pd.DataFrame(
        labor_by_state.round(2),
        index=[s[:12] for s in sectors],
        columns=state_names
    )
    try:
        import seaborn as sns
        sns.heatmap(df_heat, annot=True, fmt=".2f", cmap="RdYlGn",
                    ax=ax2, linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "Triệu lao động"})
    except Exception:
        im = ax2.imshow(labor_by_state, cmap="RdYlGn", aspect="auto")
        ax2.set_xticks(range(4)); ax2.set_xticklabels(state_names, rotation=30)
        ax2.set_yticks(range(8)); ax2.set_yticklabels([s[:12] for s in sectors])
        plt.colorbar(im, ax=ax2, label="Triệu lao động")
        for i in range(8):
            for j in range(4):
                ax2.text(j, i, f"{labor_by_state[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax2.set_title("Số Lao động Steady-State (triệu người)\ntheo Ngành × Trạng thái",
                  fontweight="bold")
    ax2.tick_params(axis="x", rotation=25, labelsize=9)
    ax2.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bai09_extension_markov_labor.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  Steady-state dài hạn (% lao động mỗi ngành):")
    print(f"  {'Ngành':<18} {'Ổn định':>10} {'Mất việc':>10} {'Đào tạo':>10} {'Việc mới AI':>12}")
    print("  " + "-"*55)
    for i, s in enumerate(sectors):
        print(f"  {s:<18} {ss[i,0]*100:>9.1f}% {ss[i,1]*100:>9.1f}% "
              f"{ss[i,2]*100:>9.1f}% {ss[i,3]*100:>11.1f}%")
    worst = sectors[np.argmax(ss[:, 1])]
    print(f"\n  → Ngành có % mất việc steady-state cao nhất: {worst} ({ss[:,1].max()*100:.1f}%)")
    print(f"  → Biểu đồ đã lưu: outputs/bai09_extension_markov_labor.png")


# ══════════════════════════════════════════════════════════════════
# BÀI 10 — Stochastic: Scenario Tree + EVPI Sensitivity
# ══════════════════════════════════════════════════════════════════

def bai10_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 10: Fan Chart Kịch bản + EVPI Sensitivity Analysis

    Ý nghĩa kinh tế: Fan Chart (biểu đồ quạt) trực quan hoá bất định theo thời
    gian — định dạng chuẩn của IMF/World Bank khi công bố dự báo. EVPI sensitivity
    cho thấy mức thông tin nào (về kịch bản nào) có giá trị nhất — hướng dẫn
    Chính phủ nên đầu tư vào hệ thống cảnh báo sớm loại rủi ro nào.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 10 ---")
    print("Fan Chart GDP 5 năm + EVPI Sensitivity theo kịch bản")

    # Tham số kịch bản (từ bài 10)
    scenarios    = ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"]
    probs        = [0.30, 0.45, 0.20, 0.05]
    beta_gdp_s   = [1.55, 1.25, 0.90, 0.55]  # hệ số β_AI kịch bản (proxy tăng trưởng)
    Y0           = 12847.6  # GDP 2025

    # Mô phỏng quỹ đạo GDP 2026-2030 theo kịch bản
    np.random.seed(42)
    T = 5
    years = np.arange(2026, 2031)

    # First-stage investment (tối ưu SP)
    x_sp = np.array([32500, 16250, 11375, 5875])  # nghìn tỷ VND (proxy)

    # GDP trajectory mỗi kịch bản
    gdp_trajs = {}
    for s, (sname, p, beta_s) in enumerate(zip(scenarios, probs, beta_gdp_s)):
        gdp = np.zeros(T)
        gdp[0] = Y0 * (1 + 0.005 * beta_s + 0.001 * x_sp.sum() / Y0)
        for t in range(1, T):
            shock = np.random.normal(0, 0.015 * (1 + 0.3*s), 200).mean()  # 200 mô phỏng/kịch bản
            gdp[t] = gdp[t-1] * (1 + 0.05 * beta_s + shock)
        gdp_trajs[sname] = gdp

    # Monte Carlo cho Fan Chart: toàn bộ mix kịch bản
    N_SIM = 3000
    gdp_mc = np.zeros((N_SIM, T))
    for sim in range(N_SIM):
        # Chọn kịch bản theo xác suất
        s_idx = np.random.choice(4, p=probs)
        sname = scenarios[s_idx]; beta_s = beta_gdp_s[s_idx]
        gdp_mc[sim, 0] = Y0 * (1 + 0.05 * beta_s + np.random.normal(0, 0.01))
        for t in range(1, T):
            g = 0.05 * beta_s + np.random.normal(0, 0.02)
            gdp_mc[sim, t] = gdp_mc[sim, t-1] * (1 + g)

    # Percentiles cho fan chart
    p5  = np.percentile(gdp_mc, 5, axis=0)
    p15 = np.percentile(gdp_mc, 15, axis=0)
    p25 = np.percentile(gdp_mc, 25, axis=0)
    p50 = np.percentile(gdp_mc, 50, axis=0)
    p75 = np.percentile(gdp_mc, 75, axis=0)
    p85 = np.percentile(gdp_mc, 85, axis=0)
    p95 = np.percentile(gdp_mc, 95, axis=0)

    # EVPI sensitivity: giá trị biết trước từng kịch bản
    # EVPI_s = p_s * (Z*(s) - Z_sp) với Z*(s) = optimal dưới kịch bản s
    z_sp = sum(p * b * x_sp.sum() / 1000 for p, b in zip(probs, beta_gdp_s))
    z_stars_s = [b * x_sp.sum() / 1000 for b in beta_gdp_s]  # perfect info từng s
    evpi_parts = [p * max(0, z - z_sp) for p, z in zip(probs, z_stars_s)]
    evpi_total = sum(evpi_parts)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Bài 10 — Extension: Fan Chart GDP 2026–2030 + EVPI Sensitivity\n"
                 "Chuẩn định dạng IMF/World Bank cho Báo cáo Chính sách",
                 fontsize=13, fontweight="bold")

    # Panel 1: Fan Chart
    ax = axes[0]
    ax.fill_between(years, p5,  p95, alpha=0.12, color="#1565C0", label="Khoảng 90%")
    ax.fill_between(years, p15, p85, alpha=0.18, color="#1565C0", label="Khoảng 70%")
    ax.fill_between(years, p25, p75, alpha=0.28, color="#1565C0", label="Khoảng 50%")
    ax.plot(years, p50, color="#1565C0", lw=3, label="Trung vị (P50)")
    # Quỹ đạo từng kịch bản
    colors_scen = ["#2E7D32", "#F59E0B", "#E65100", "#C62828"]
    for sname, color in zip(scenarios, colors_scen):
        ax.plot(years, gdp_trajs[sname], "--", color=color, lw=1.5,
                alpha=0.8, label=sname)
    # GDP 2025 anchor
    ax.scatter([2025], [Y0], s=120, color="black", zorder=10)
    ax.annotate(f"GDP 2025\n={Y0:,.0f}", (2025, Y0),
                textcoords="offset points", xytext=(10, -20), fontsize=8)
    ax.axhline(Y0 * (1.07**5), color="red", ls=":", lw=1.5,
               label=f"Mục tiêu 7%/năm = {Y0*(1.07**5):,.0f}")
    ax.set_xlabel("Năm", fontsize=11)
    ax.set_ylabel("GDP (nghìn tỷ VND)", fontsize=11)
    ax.set_title("Fan Chart GDP 2026–2030\n(Hai giai đoạn Stochastic Programming)",
                 fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left"); ax.grid(alpha=0.3)
    ax.set_xlim(2024.5, 2030.5)

    # Panel 2: EVPI decomposition
    ax2 = axes[1]
    bar_colors = ["#2E7D32", "#F59E0B", "#E65100", "#C62828"]
    bars = ax2.bar(scenarios, evpi_parts, color=bar_colors, alpha=0.85,
                   edgecolor="white", width=0.5)
    ax2.axhline(evpi_total, color="#1565C0", ls="--", lw=2,
                label=f"EVPI tổng = {evpi_total:.3f}")
    for bar, val in zip(bars, evpi_parts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.3f}", ha="center", fontweight="bold", fontsize=10)
    ax2.set_ylabel("Đóng góp vào EVPI (nghìn tỷ VND)", fontsize=10)
    ax2.set_title("EVPI Decomposition theo Kịch bản\n"
                  "(Kịch bản nào cần biết trước nhất?)", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")
    note = f"EVPI = {evpi_total:.3f} nghìn tỷ → Giá trị tối đa\ntrả cho hệ thống dự báo hoàn hảo"
    ax2.text(0.5, 0.95, note, transform=ax2.transAxes, ha="center", va="top",
             fontsize=9, bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bai10_extension_fan_chart_evpi.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  → EVPI tổng: {evpi_total:.4f} nghìn tỷ VND")
    print(f"  → Kịch bản đóng góp EVPI nhiều nhất: {scenarios[np.argmax(evpi_parts)]}"
          f" ({max(evpi_parts):.4f})")
    print(f"  → GDP P50 năm 2030: {p50[-1]:,.1f} | P5: {p5[-1]:,.1f} | P95: {p95[-1]:,.1f}")
    print(f"  → Biểu đồ đã lưu: outputs/bai10_extension_fan_chart_evpi.png")


# ══════════════════════════════════════════════════════════════════
# BÀI 11 — Q-Learning: Q-Table Heatmap + Learning Curve Convergence
# ══════════════════════════════════════════════════════════════════

def bai11_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 11: Q-Table Heatmap + Convergence Analysis + Policy Map

    Ý nghĩa kinh tế: Trực quan hoá Q-Table cho phép "mở hộp đen" của Q-Learning —
    giải thích TẠI SAO agent chọn hành động A tại trạng thái S. Convergence plot
    và policy map toàn trạng thái là công cụ báo cáo chuẩn trong RL nghiên cứu.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 11 ---")
    print("Q-Table Heatmap + Convergence + Policy Map toàn trạng thái")

    # ── Tái định nghĩa môi trường tối giản (không phụ thuộc file gốc) ──
    ALLOCATIONS = {
        0: np.array([0.70, 0.10, 0.10, 0.10]),
        1: np.array([0.40, 0.25, 0.15, 0.20]),
        2: np.array([0.25, 0.45, 0.15, 0.15]),
        3: np.array([0.20, 0.20, 0.45, 0.15]),
        4: np.array([0.30, 0.20, 0.10, 0.40]),
    }
    ACTION_NAMES = ["Truyền thống", "Cân bằng", "Số hóa nhanh", "AI dẫn dắt", "Bao trùm số"]
    N_STATES  = 81   # 3^4
    N_ACTIONS = 5
    ALPHA, GAMMA, EPS_START, EPS_END = 0.12, 0.92, 1.0, 0.05
    N_EPISODES = 8000

    def state_to_tuple(s):
        """Chuyển state index → (gdp_lvl, d_lvl, ai_lvl, unemp_lvl) mỗi ∈ {0,1,2}."""
        indices = []
        for _ in range(4):
            indices.append(s % 3); s //= 3
        return tuple(indices)

    def compute_reward(alloc, state_tup):
        gdp_l, d_l, ai_l, u_l = state_tup
        w_K, w_D, w_AI, w_H = alloc
        gdp_base = 6 + gdp_l*2
        delta_gdp = (w_K*3.3 + w_D*1.0 + w_AI*0.8 + w_H*0.7) * (1 + 0.1*(d_l + ai_l))
        delta_u   = -0.03*w_AI + 0.02*w_H - 0.01*(u_l - 1)
        cyber     = 0.08*w_AI + 0.05*w_K
        emission  = 0.12*w_K + 0.06*w_D
        r = (0.40*delta_gdp/gdp_base - 0.25*abs(delta_u) - 0.20*cyber - 0.15*emission)
        return float(r)

    def step(state, action):
        tup     = state_to_tuple(state)
        alloc   = ALLOCATIONS[action]
        reward  = compute_reward(alloc, tup)
        # Transition: random walk với bias từ action
        gdp_l, d_l, ai_l, u_l = tup
        gdp_new = int(np.clip(gdp_l + np.random.choice([-1,0,0,1], p=[0.1,0.4,0.4,0.1])
                              + (1 if alloc[0]>0.4 else 0), 0, 2))
        d_new   = int(np.clip(d_l  + (1 if alloc[1]>0.3 else 0) + np.random.choice([-1,0,0], p=[0.1,0.5,0.4]), 0, 2))
        ai_new  = int(np.clip(ai_l + (1 if alloc[2]>0.3 else 0) + np.random.choice([-1,0,0], p=[0.1,0.5,0.4]), 0, 2))
        u_new   = int(np.clip(u_l  + (1 if alloc[3]<0.15 else -1) + np.random.choice([-1,0,1], p=[0.3,0.4,0.3]), 0, 2))
        new_state = gdp_new + d_new*3 + ai_new*9 + u_new*27
        return new_state, reward

    # ── Q-Learning training ─────────────────────────────────────
    Q  = np.zeros((N_STATES, N_ACTIONS))
    episode_rewards = []
    eps = EPS_START

    for ep in range(N_EPISODES):
        state = np.random.randint(0, N_STATES)
        ep_r  = 0
        for _ in range(15):  # 15 bước/episode
            if np.random.random() < eps:
                action = np.random.randint(0, N_ACTIONS)
            else:
                action = np.argmax(Q[state])
            new_state, reward = step(state, action)
            Q[state, action] += ALPHA * (reward + GAMMA * Q[new_state].max() - Q[state, action])
            state = new_state; ep_r += reward
        episode_rewards.append(ep_r)
        eps = max(EPS_END, eps * 0.9995)

    print(f"  Q-Learning hoàn thành {N_EPISODES:,} episodes | ε cuối: {eps:.4f}")

    # ── Policy π*(s) toàn bộ states ────────────────────────────
    pi_star = Q.argmax(axis=1)

    # ── Tạo biểu đồ ─────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Bài 11 — Extension: Q-Table Heatmap + Convergence + Policy Map\n"
                 "Giải thích Q-Learning Agent: TẠI SAO chọn hành động A tại trạng thái S?",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Q-Table heatmap (81×5)
    ax1 = fig.add_subplot(gs[0, :2])
    try:
        import seaborn as sns
        # Hiển thị Q-table normalised per state
        Q_norm = Q - Q.min(axis=1, keepdims=True)
        Q_norm /= (Q_norm.max(axis=1, keepdims=True) + 1e-9)
        sns.heatmap(Q_norm[:40].T, cmap="YlOrRd", ax=ax1,
                    xticklabels=range(0, 40), yticklabels=ACTION_NAMES,
                    linewidths=0.2, linecolor="white",
                    cbar_kws={"label": "Q-value normalised"})
    except Exception:
        im = ax1.imshow(Q[:40].T, cmap="YlOrRd", aspect="auto")
        ax1.set_yticks(range(5)); ax1.set_yticklabels(ACTION_NAMES)
        plt.colorbar(im, ax=ax1)
    ax1.set_xlabel("State index (0–39 trong 81 states)", fontsize=10)
    ax1.set_title("Q-Table Heatmap (40 states × 5 actions) — Giá trị Q chuẩn hóa",
                  fontweight="bold")
    ax1.tick_params(axis="x", labelsize=7)

    # Panel 2: Learning curve (rolling mean)
    ax2 = fig.add_subplot(gs[0, 2])
    window = 200
    roll_mean = pd.Series(episode_rewards).rolling(window).mean()
    ax2.plot(range(N_EPISODES), episode_rewards, alpha=0.15, color="#B0BEC5", lw=0.5)
    ax2.plot(range(N_EPISODES), roll_mean, color="#1565C0", lw=2.5,
             label=f"Rolling mean ({window} ep)")
    ax2.axhline(roll_mean.dropna().iloc[-100:].mean(), color="red", ls="--", lw=2,
                label=f"Converged ≈ {roll_mean.dropna().iloc[-100:].mean():.3f}")
    ax2.set_xlabel("Episode", fontsize=10); ax2.set_ylabel("Reward tổng", fontsize=10)
    ax2.set_title("Learning Curve\nHội tụ Q-Learning", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Panel 3: Policy Map — heatmap π*(s) theo (GDP, D) với AI=1, U=1
    ax3 = fig.add_subplot(gs[1, 0])
    policy_grid = np.zeros((3, 3))
    for gdp_l in range(3):
        for d_l in range(3):
            s = gdp_l + d_l*3 + 1*9 + 1*27
            policy_grid[gdp_l, d_l] = pi_star[s]
    im3 = ax3.imshow(policy_grid, cmap="tab10", vmin=0, vmax=4, aspect="auto")
    ax3.set_xticks(range(3)); ax3.set_xticklabels(["D Thấp", "D Trung", "D Cao"])
    ax3.set_yticks(range(3)); ax3.set_yticklabels(["GDP Thấp", "GDP Trung", "GDP Cao"])
    for gdp_l in range(3):
        for d_l in range(3):
            act_idx = int(policy_grid[gdp_l, d_l])
            ax3.text(d_l, gdp_l, ACTION_NAMES[act_idx][:8],
                     ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
    plt.colorbar(im3, ax=ax3, ticks=range(5),
                 label="Hành động tối ưu")
    ax3.set_title("Policy Map π*(s)\n(AI=Trung, Unemp=Trung)", fontweight="bold")

    # Panel 4: Action frequency histogram
    ax4 = fig.add_subplot(gs[1, 1])
    action_counts = np.bincount(pi_star, minlength=5)
    colors_act = ["#6B7280", "#F59E0B", "#3B82F6", "#8B5CF6", "#10B981"]
    bars = ax4.bar(ACTION_NAMES, action_counts, color=colors_act, alpha=0.85,
                   edgecolor="white")
    for bar, cnt in zip(bars, action_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{cnt}\n({cnt/81*100:.0f}%)", ha="center", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Số trạng thái được chọn", fontsize=10)
    ax4.set_title("Tần suất hành động trong π*(s)\n(81 trạng thái)", fontweight="bold")
    ax4.tick_params(axis="x", rotation=25, labelsize=9); ax4.grid(alpha=0.3, axis="y")

    # Panel 5: Q-value advantage A(s,a) = Q(s,a) - V(s)
    ax5 = fig.add_subplot(gs[1, 2])
    V_s = Q.max(axis=1)
    A_sa = Q - V_s[:, None]  # advantage
    # Trung bình advantage theo action
    adv_mean = A_sa.mean(axis=0)
    adv_std  = A_sa.std(axis=0)
    ax5.bar(ACTION_NAMES, adv_mean, yerr=adv_std, color=colors_act, alpha=0.85,
            edgecolor="white", capsize=5)
    ax5.axhline(0, color="gray", lw=0.8)
    ax5.set_ylabel("Advantage A(s,a) = Q(s,a) − V(s)", fontsize=9)
    ax5.set_title("Advantage Function Trung bình\ntheo Hành động", fontweight="bold")
    ax5.tick_params(axis="x", rotation=25, labelsize=9); ax5.grid(alpha=0.3, axis="y")

    plt.savefig(OUTPUT_DIR / "bai11_extension_qtable_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Phân phối Policy π*: {dict(zip(ACTION_NAMES, action_counts.tolist()))}")
    dominant = ACTION_NAMES[action_counts.argmax()]
    print(f"  → Hành động phổ biến nhất: '{dominant}' ({action_counts.max()} trạng thái = "
          f"{action_counts.max()/81*100:.0f}%)")
    print(f"  → Biểu đồ đã lưu: outputs/bai11_extension_qtable_analysis.png")


# ══════════════════════════════════════════════════════════════════
# BÀI 12 — AIDEOM-VN: Integrated Policy Scorecard Dashboard
# ══════════════════════════════════════════════════════════════════

def bai12_creative_extension():
    """
    MỞ RỘNG SÁNG TẠO BÀI 12: Integrated Policy Scorecard Dashboard

    Ý nghĩa kinh tế: Tổng hợp KPI từ TẤT CẢ 6 module (M1–M6) thành một
    Policy Scorecard dạng Balanced Scorecard — công cụ báo cáo chuẩn cho
    cơ quan hoạch định chính sách. Dashboard này cho phép so sánh nhanh
    5 kịch bản trên 12 chiều KPI, kết hợp radar chart đa chiều và
    traffic-light alerting tự động (xanh/vàng/đỏ theo ngưỡng chính sách).
    Đây là "lớp tổng hợp" nối toàn bộ pipeline M1→M5.
    """
    print("\n--- MỞ RỘNG SÁNG TẠO BÀI 12 ---")
    print("Integrated Policy Scorecard: 5 Kịch bản × 12 KPI")
    print("Traffic-light alerting + Radar Chart + Heatmap tổng hợp")

    # ═══════════════════════════════════════════════════════
    # Dữ liệu KPI tổng hợp từ pipeline M1–M5
    # (dùng kết quả mô phỏng để đảm bảo hoạt động độc lập)
    # ═══════════════════════════════════════════════════════
    scenarios = ["S1\nTruyền thống", "S2\nSố hóa nhanh",
                 "S3\nAI dẫn dắt", "S4\nBao trùm số", "S5\nCân bằng tối ưu"]
    sid       = ["S1", "S2", "S3", "S4", "S5"]

    kpi_labels = [
        "GDP growth\n(%/năm)",
        "GDP 2030\n(nghìn tỷ)",
        "Kinh tế số\n(%GDP)",
        "AI firms\n(nghìn DN)",
        "Nhân lực số\n(%)",
        "NetJob\n(nghìn việc)",
        "VaR 95%\n(nghìn tỷ)",
        "Xác suất đạt\nmục tiêu (%)",
        "Z* LP\n(tỷ VND)",
        "Equity\n(Gini ↓)",
        "TFP tăng\n(%/năm)",
        "TOPSIS top3\nổn định",
    ]

    # KPI matrix [5 kịch bản × 12 KPI]  (đơn vị chuẩn hoá hóa trong từng cột)
    kpi_raw = np.array([
        # S1: Truyền thống
        [6.21,  15200, 21.0,  92,  31.0,  980,  13800, 52.0,  61200, 0.38, 0.8, 0.67],
        # S2: Số hóa nhanh
        [7.15,  16800, 28.5,  98,  33.5, 1150,  14200, 64.0,  63800, 0.34, 1.2, 0.78],
        # S3: AI dẫn dắt
        [7.42,  17400, 25.0, 112,  32.0, 1080,  14000, 66.0,  64100, 0.35, 1.5, 0.72],
        # S4: Bao trùm số
        [6.85,  16200, 23.0,  94,  36.5, 1310,  13900, 60.0,  62500, 0.30, 1.0, 0.75],
        # S5: Cân bằng tối ưu
        [7.28,  17100, 26.5, 102,  34.5, 1315,  14500, 69.0,  64196, 0.32, 1.3, 0.85],
    ])

    # Ngưỡng chính sách (targets từ Nghị quyết 57, QĐ 749)
    targets = [7.0,  17000, 30.0, 100, 35.0, 1200, 14000, 65.0, 63000, 0.33, 1.0, 0.75]
    # Hướng tốt: True = cao hơn = tốt hơn
    higher_better = [True, True, True, True, True, True, True, True, True, False, True, True]

    # ── Tính điểm chuẩn hóa [0, 1] ─────────────────────────────
    kpi_norm = np.zeros_like(kpi_raw)
    for j in range(12):
        col = kpi_raw[:, j]
        mn, mx = col.min(), col.max()
        if mx > mn:
            if higher_better[j]:
                kpi_norm[:, j] = (col - mn) / (mx - mn)
            else:
                kpi_norm[:, j] = (mx - col) / (mx - mn)
        else:
            kpi_norm[:, j] = 0.5

    # ── Traffic light: xanh/vàng/đỏ theo ngưỡng ────────────────
    traffic = np.zeros_like(kpi_raw, dtype=int)  # 0=đỏ, 1=vàng, 2=xanh
    for i in range(5):
        for j in range(12):
            v, t = kpi_raw[i, j], targets[j]
            ratio = v / t if t != 0 else 1.0
            if higher_better[j]:
                traffic[i, j] = 2 if ratio >= 0.95 else (1 if ratio >= 0.80 else 0)
            else:
                traffic[i, j] = 2 if ratio <= 1.05 else (1 if ratio <= 1.20 else 0)

    # ── Overall score ─────────────────────────────────────────
    overall = kpi_norm.mean(axis=1)

    # ════════════════════════════════
    # Vẽ Dashboard
    # ════════════════════════════════
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle(
        "AIDEOM-VN — Integrated Policy Scorecard Dashboard\n"
        "5 Kịch bản × 12 KPI | Tổng hợp M1→M6 | Bài 12 Extension",
        fontsize=15, fontweight="bold", y=0.98, color="#1E293B"
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.40,
                           top=0.93, bottom=0.04)

    # ── Panel 1: KPI Heatmap tổng hợp (traffic-light) ───────────
    ax1 = fig.add_subplot(gs[0, :2])
    traffic_colors = {0: "#FEE2E2", 1: "#FEF3C7", 2: "#D1FAE5"}
    border_colors  = {0: "#EF4444", 1: "#F59E0B", 2: "#10B981"}

    ax1.set_xlim(-0.5, 11.5); ax1.set_ylim(-0.5, 4.5)
    ax1.set_facecolor("#F1F5F9")
    for i in range(5):
        for j in range(12):
            fc = traffic_colors[traffic[i, j]]
            ec = border_colors[traffic[i, j]]
            rect = plt.Rectangle((j-0.45, (4-i)-0.45), 0.9, 0.9,
                                  facecolor=fc, edgecolor=ec, lw=1.5)
            ax1.add_patch(rect)
            val = kpi_raw[i, j]
            txt = f"{val:.1f}" if val < 1000 else f"{val/1000:.1f}k"
            ax1.text(j, 4-i, txt, ha="center", va="center",
                     fontsize=8, fontweight="bold", color="#1E293B")
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(kpi_labels, fontsize=7.5, rotation=0)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(reversed(["S1 Truyền thống", "S2 Số hóa nhanh",
                                   "S3 AI dẫn dắt", "S4 Bao trùm số",
                                   "S5 Cân bằng"]), fontsize=9, fontweight="bold")
    ax1.set_title("KPI Scorecard — Traffic-Light Alerting\n"
                  "🟢 ≥95% mục tiêu   🟡 80–95%   🔴 <80%",
                  fontweight="bold", fontsize=11)
    ax1.grid(False)
    # Chú thích màu
    for label, color, ec in [("Đạt mục tiêu", "#D1FAE5", "#10B981"),
                              ("Cần chú ý", "#FEF3C7", "#F59E0B"),
                              ("Dưới ngưỡng", "#FEE2E2", "#EF4444")]:
        rect = plt.Rectangle((0, 0), 1, 1, fc=color, ec=ec, lw=1.5)
        ax1.add_patch(plt.Rectangle((0,0), 0, 0, fc=color, ec=ec, label=label))
    ax1.legend(loc="upper right", fontsize=8,
               handles=[plt.Rectangle((0,0), 1, 1, fc=c, ec=e, label=l)
                         for l, c, e in [("Đạt ≥95%", "#D1FAE5", "#10B981"),
                                          ("Cần chú ý 80–95%", "#FEF3C7", "#F59E0B"),
                                          ("Dưới ngưỡng", "#FEE2E2", "#EF4444")]])

    # ── Panel 2: Overall Score Ranking ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    sort_idx  = np.argsort(overall)[::-1]
    bar_colors_ov = ["#10B981", "#3B82F6", "#8B5CF6", "#F59E0B", "#6B7280"]
    bars = ax2.barh(range(5), overall[sort_idx],
                    color=[bar_colors_ov[i] for i in sort_idx],
                    alpha=0.9, edgecolor="white", height=0.6)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels([sid[i] for i in sort_idx], fontsize=10, fontweight="bold")
    ax2.set_xlabel("Overall Score (chuẩn hóa 0–1)", fontsize=10)
    ax2.set_title("Xếp hạng Tổng thể\n12 KPI Chuẩn hóa", fontweight="bold")
    for bar, val in zip(bars, overall[sort_idx]):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontweight="bold", fontsize=10)
    ax2.set_xlim(0, 1.1); ax2.grid(alpha=0.3, axis="x")
    ax2.axvline(0.7, color="red", ls="--", lw=1.5, alpha=0.7, label="Ngưỡng 0.70")
    ax2.legend(fontsize=8)

    # ── Panel 3–4: Radar Chart 2 kịch bản tốt nhất ─────────────
    best2 = np.argsort(overall)[::-1][:2]
    radar_dims  = kpi_labels
    N_r = len(radar_dims)
    angles_r = np.linspace(0, 2*np.pi, N_r, endpoint=False).tolist()
    angles_r += angles_r[:1]
    radar_colors = ["#10B981", "#3B82F6", "#8B5CF6", "#F59E0B", "#6B7280"]

    ax_radar = fig.add_subplot(gs[1, :2], polar=True)
    for i in range(5):
        vals = kpi_norm[i].tolist() + [kpi_norm[i, 0]]
        alpha_v = 0.8 if i in best2 else 0.25
        lw_v    = 2.5 if i in best2 else 0.8
        ax_radar.plot(angles_r, vals, color=radar_colors[i],
                      lw=lw_v, alpha=alpha_v, label=sid[i])
        if i in best2:
            ax_radar.fill(angles_r, vals, color=radar_colors[i], alpha=0.08)
    ax_radar.set_thetagrids(np.degrees(angles_r[:-1]),
                            [k.replace("\n", " ")[:15] for k in kpi_labels], fontsize=7)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Radar Chart — 12 KPI Chuẩn hóa\n(Highlight 2 kịch bản tốt nhất)",
                        fontweight="bold", pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax_radar.grid(alpha=0.4)

    # ── Panel 5: KPI Gap Analysis (vs targets) ──────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    gap_s5 = np.zeros(12)
    for j in range(12):
        t = targets[j]
        v = kpi_raw[4, j]  # S5
        if higher_better[j]:
            gap_s5[j] = (v - t) / t * 100
        else:
            gap_s5[j] = (t - v) / t * 100
    colors_gap = ["#10B981" if g >= 0 else "#EF4444" for g in gap_s5]
    short_labels = [k.split("\n")[0] for k in kpi_labels]
    bars5 = ax5.barh(range(12), gap_s5, color=colors_gap, alpha=0.85, edgecolor="white")
    ax5.axvline(0, color="gray", lw=1)
    ax5.set_yticks(range(12)); ax5.set_yticklabels(short_labels, fontsize=8)
    ax5.set_xlabel("Gap so với mục tiêu (%)", fontsize=9)
    ax5.set_title("Gap Analysis — S5 vs Mục tiêu\n(+% = vượt | -% = thiếu)",
                  fontweight="bold")
    ax5.grid(alpha=0.3, axis="x")

    # ── Panel 6: Timeline KPI S5 (giả lập M1 forecast) ──────────
    ax6 = fig.add_subplot(gs[2, :2])
    years_tl = np.array([2025, 2026, 2027, 2028, 2029, 2030])
    gdp_s5  = np.array([12847.6, 13720, 14680, 15510, 16380, 17100])
    d_s5    = np.array([19.5, 21.5, 23.0, 24.5, 25.8, 26.5])
    ai_s5   = np.array([80.1, 85, 90, 95, 99, 102])
    h_s5    = np.array([29.2, 30.5, 31.8, 33.0, 34.0, 34.5])

    ax6_twin = ax6.twinx()
    l1, = ax6.plot(years_tl, gdp_s5, "o-", color="#1565C0", lw=2.5, ms=7, label="GDP S5")
    l2, = ax6_twin.plot(years_tl, d_s5, "s--", color="#E65100", lw=2, ms=6, label="D% GDP")
    l3, = ax6_twin.plot(years_tl, h_s5, "^-.", color="#2E7D32", lw=2, ms=6, label="H% lao động")
    ax6.fill_between(years_tl, gdp_s5, alpha=0.08, color="#1565C0")
    ax6.set_xlabel("Năm", fontsize=10)
    ax6.set_ylabel("GDP (nghìn tỷ VND)", color="#1565C0", fontsize=10)
    ax6_twin.set_ylabel("D% / H% (%)", color="#E65100", fontsize=10)
    ax6.set_title("Lộ trình KPI Kịch bản S5 (Cân bằng tối ưu) 2025→2030",
                  fontweight="bold")
    ax6.axvline(2030, color="red", ls=":", lw=1.5, alpha=0.7)
    # Mục tiêu 2030
    ax6.axhline(17000, color="#1565C0", ls=":", lw=1, alpha=0.5)
    ax6_twin.axhline(30, color="#E65100", ls=":", lw=1, alpha=0.5)
    ax6.legend(handles=[l1, l2, l3], fontsize=9, loc="upper left")
    ax6.grid(alpha=0.3)
    ax6.tick_params(axis="y", labelcolor="#1565C0")
    ax6_twin.tick_params(axis="y", labelcolor="#E65100")

    # ── Panel 7: Policy Recommendation Box ──────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    best_sid  = sid[np.argmax(overall)]
    worst_kpi = kpi_labels[np.argmin(kpi_norm[4])].replace("\n", " ")
    green_count = (traffic == 2).sum(axis=1)
    red_count   = (traffic == 0).sum(axis=1)

    rec_text = (
        "📊 POLICY RECOMMENDATION\n"
        "══════════════════════════════\n\n"
        f"🏆 Kịch bản tổng thể tốt nhất:\n"
        f"   {best_sid} (Overall={overall.max():.3f})\n\n"
        "📈 KPI đạt mục tiêu (xanh):\n"
        + "".join(f"   {sid[i]}: {green_count[i]}/12 KPI\n" for i in range(5)) +
        "\n❌ Điểm yếu S5 nhất:\n"
        f"   {worst_kpi}\n\n"
        "🎯 Khuyến nghị:\n"
        "   1. Ưu tiên S5 (Cân bằng tối ưu)\n"
        "      → GDP + Công bằng tốt nhất\n"
        "   2. Bổ sung chính sách nhân lực\n"
        "      từ S4 vào S5\n"
        "   3. Review equity λ nếu Gini > 0.35\n"
        "   4. Trigger cảnh báo nếu VaR < 13,500\n\n"
        "📋 Nguồn: M1(TFP) + M2(TOPSIS)\n"
        "   + M3(LP) + M4(Labor) + M5(MC)"
    )
    ax7.text(0.03, 0.97, rec_text, transform=ax7.transAxes,
             fontsize=8, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#EFF6FF",
                       edgecolor="#1565C0", alpha=0.9, lw=1.5))

    plt.savefig(OUTPUT_DIR / "bai12_extension_policy_scorecard.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  ══ KPI SCORECARD TÓM TẮT ══")
    print(f"  {'Kịch bản':<20} {'Overall Score':>14} {'Xanh/12':>9} {'Đỏ/12':>8}")
    print("  " + "-"*55)
    for i in np.argsort(overall)[::-1]:
        g = (traffic[i] == 2).sum()
        r = (traffic[i] == 0).sum()
        print(f"  {sid[i]+' '+scenarios[i].split()[1]:<20} "
              f"{overall[i]:>13.4f} {g:>9} {r:>8}")
    print(f"\n  → Kịch bản tổng thể tốt nhất: {sid[np.argmax(overall)]}"
          f" (Overall={overall.max():.4f})")
    print(f"  → Kịch bản kém nhất về KPI: {sid[np.argmin(overall)]}")
    print(f"  → Biểu đồ đã lưu: outputs/bai12_extension_policy_scorecard.png")


# ══════════════════════════════════════════════════════════════════
# CHẠY TẤT CẢ
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  AIDEOM-VN — MỞ RỘNG SÁNG TẠO TẤT CẢ 12 BÀI")
    print("  Chạy tuần tự Bài 1 → Bài 12")
    print("=" * 65)

    extensions = [
        ("BÀI 1",  bai1_creative_extension),
        ("BÀI 2",  bai2_creative_extension),
        ("BÀI 3",  bai3_creative_extension),
        ("BÀI 4",  bai4_creative_extension),
        ("BÀI 5",  bai5_creative_extension),
        ("BÀI 6",  bai6_creative_extension),
        ("BÀI 7",  bai7_creative_extension),
        ("BÀI 8",  bai8_creative_extension),
        ("BÀI 9",  bai9_creative_extension),
        ("BÀI 10", bai10_creative_extension),
        ("BÀI 11", bai11_creative_extension),
        ("BÀI 12", bai12_creative_extension),
    ]

    errors = {}
    for name, fn in extensions:
        print(f"\n{'='*65}")
        print(f"  ĐANG CHẠY: {name}")
        print(f"{'='*65}")
        try:
            fn()
        except Exception as e:
            import traceback
            errors[name] = str(e)
            print(f"  ✗ LỖI: {e}")
            traceback.print_exc()

    print("\n" + "="*65)
    print("  HOÀN THÀNH TẤT CẢ EXTENSIONS")
    saved = list((OUTPUT_DIR).glob("ba*_extension_*.png"))
    print(f"  Tổng file đã lưu: {len(saved)}")
    for f in sorted(saved):
        print(f"    ✓ {f.name}")
    if errors:
        print(f"\n  Có {len(errors)} lỗi:")
        for name, err in errors.items():
            print(f"    {name}: {err}")
    print("="*65)
