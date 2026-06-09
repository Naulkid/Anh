"""
Bài 7: Tối ưu đa mục tiêu Pareto với NSGA-II
==============================================
Bốn mục tiêu:
  f1 = max GDP gain (Σ β_jr * x_jr)
  f2 = min Gini bất bình đẳng vùng (MAD)
  f3 = min phát thải (Σ e_r*(x_I,r + x_AI,r))
  f4 = min rủi ro an ninh dữ liệu ròng (Σ ρ_r*x_AI,r - σ_r*x_H,r)

24 biến quyết định: x[6 vùng × 4 hạng mục (I,D,AI,H)]
Ràng buộc giống Bài 4 (C1-C4) + lỏng C5

Yêu cầu:
    7.4.1 - NSGA-II, pop=100, gen=200
    7.4.2 - Scatter 3D Pareto + parallel coordinates
    7.4.3 - TOPSIS chọn nghiệm thoả hiệp
    7.4.4 - Chi phí cơ hội giữa các nghiệm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    PYMOO_OK = True
except ImportError:
    PYMOO_OK = False
    print("⚠ pymoo chưa cài — pip install pymoo")

print("=" * 60)
print("BÀI 7: TỐI ƯU ĐA MỤC TIÊU PARETO — NSGA-II")
print("=" * 60)

# ─────────────────────────────────────────
# THAM SỐ
# ─────────────────────────────────────────
region_names = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']

# β matrix [6 × 4]: cột = I, D, AI, H
beta = np.array([
    [1.15, 0.85, 0.55, 1.30],
    [0.95, 1.25, 1.40, 1.05],
    [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35],
    [0.90, 1.30, 1.55, 1.00],
    [1.10, 0.85, 0.65, 1.25],
])

e   = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])  # CO2/tỷ
rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])  # rủi ro AI/tỷ
sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])  # giảm rủi ro/H

BUDGET_TOTAL = 50000
FLOOR_REGION = 5000
CEIL_REGION  = 12000
FLOOR_HUMAN  = 12000

# ─────────────────────────────────────────
# ĐỊNH NGHĨA BÀI TOÁN PYMOO
# ─────────────────────────────────────────
if PYMOO_OK:
    class VietnamDigitalProblem(ElementwiseProblem):
        def __init__(self):
            # 24 biến: x[0..5] = I, x[6..11] = D, x[12..17] = AI, x[18..23] = H
            # Hoặc reshape thành (6,4): X[r, j]
            super().__init__(
                n_var=24,
                n_obj=4,
                n_ieq_constr=14,   # C1 + 6*C2 + 6*C3 + C4
                xl=np.zeros(24),
                xu=np.ones(24) * CEIL_REGION
            )

        def _evaluate(self, x, out, *args, **kwargs):
            X = x.reshape(6, 4)  # [vùng × (I,D,AI,H)]

            # --- Mục tiêu ---
            # f1: max GDP gain → minimize -f1
            f1 = -(beta * X).sum()

            # f2: Gini xấp xỉ MAD chuẩn hóa (min)
            sums = X.sum(axis=1)
            f2 = np.abs(sums - sums.mean()).mean()

            # f3: phát thải (min)
            f3 = (e * (X[:, 0] + X[:, 2])).sum()

            # f4: rủi ro ròng (min)
            f4 = (rho * X[:, 2]).sum() - (sig * X[:, 3]).sum()

            out['F'] = [f1, f2, f3, f4]

            # --- Ràng buộc g <= 0 ---
            g = []
            # C1: tổng ngân sách
            g.append(X.sum() - BUDGET_TOTAL)
            # C2: sàn mỗi vùng (6 ràng buộc)
            for r in range(6):
                g.append(FLOOR_REGION - X[r].sum())
            # C3: trần mỗi vùng (6 ràng buộc)
            for r in range(6):
                g.append(X[r].sum() - CEIL_REGION)
            # C4: sàn nhân lực tổng
            g.append(FLOOR_HUMAN - X[:, 3].sum())

            out['G'] = g

    # ─────────────────────────────────────────
    # CÂU 7.4.1: Chạy NSGA-II
    # ─────────────────────────────────────────
    print("\n--- Câu 7.4.1: Chạy NSGA-II (pop=100, gen=200) ---")

    problem = VietnamDigitalProblem()

    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    print("Đang tối ưu hoá... (có thể mất 30-60 giây)")
    result = pymoo_minimize(
        problem,
        algorithm,
        ('n_gen', 200),
        seed=42,
        verbose=False
    )

    # Lấy Pareto front
    F = result.F   # shape (n_pareto, 4)
    X_pareto = result.X  # shape (n_pareto, 24)

    # Đổi dấu f1 → GDP gain thực
    F_display = F.copy()
    F_display[:, 0] = -F_display[:, 0]

    print(f"\nSố nghiệm Pareto tìm được: {len(F)}")
    print(f"\nThống kê Pareto Front:")
    labels_f = ['f1 GDP gain', 'f2 Gini MAD', 'f3 Phát thải', 'f4 Rủi ro ròng']
    for i, lbl in enumerate(labels_f):
        col = F_display[:, i]
        print(f"  {lbl}: min={col.min():,.1f}  max={col.max():,.1f}  mean={col.mean():,.1f}")

    # ─────────────────────────────────────────
    # CÂU 7.4.2: Biểu đồ Pareto 3D + parallel coordinates
    # ─────────────────────────────────────────
    print("\n--- Câu 7.4.2: Vẽ biểu đồ Pareto ---")

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Bài 7: Tối ưu đa mục tiêu NSGA-II — Pareto Front", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Scatter 3D: f1, f2, f3 (màu theo f4)
    ax3d = fig.add_subplot(gs[0, :], projection='3d')
    sc = ax3d.scatter(
        F_display[:, 0], F_display[:, 1], F_display[:, 2],
        c=F_display[:, 3], cmap='RdYlGn_r', s=30, alpha=0.7
    )
    ax3d.set_xlabel('f1: GDP gain (tỷ VND)', fontsize=9)
    ax3d.set_ylabel('f2: Gini MAD', fontsize=9)
    ax3d.set_zlabel('f3: Phát thải', fontsize=9)
    ax3d.set_title('Pareto Front 3D (f1, f2, f3) — màu theo f4 rủi ro', fontweight='bold')
    plt.colorbar(sc, ax=ax3d, label='f4: Rủi ro ròng', shrink=0.5)

    # Parallel coordinates (tất cả 4 mục tiêu)
    ax_pc = fig.add_subplot(gs[1, :])
    F_norm = (F_display - F_display.min(axis=0)) / (F_display.max(axis=0) - F_display.min(axis=0) + 1e-9)
    # Vẽ từng nghiệm Pareto
    for i in range(len(F_norm)):
        ax_pc.plot(range(4), F_norm[i], alpha=0.15, color='#1565C0', linewidth=0.7)
    # Highlight nghiệm tốt nhất f1
    best_f1_idx = F_display[:, 0].argmax()
    ax_pc.plot(range(4), F_norm[best_f1_idx], 'r-o', linewidth=2.5, markersize=8,
               label=f'Best GDP gain (f1={F_display[best_f1_idx,0]:,.0f})')
    # Highlight nghiệm tốt nhất f2
    best_f2_idx = F_display[:, 1].argmin()
    ax_pc.plot(range(4), F_norm[best_f2_idx], 'g-s', linewidth=2.5, markersize=8,
               label=f'Best Equity (f2={F_display[best_f2_idx,1]:,.1f})')
    ax_pc.set_xticks(range(4))
    ax_pc.set_xticklabels(['f1: GDP gain↑', 'f2: Gini↓', 'f3: Phát thải↓', 'f4: Rủi ro↓'],
                           fontsize=10)
    ax_pc.set_ylabel("Giá trị chuẩn hóa [0,1]")
    ax_pc.set_title("Parallel Coordinates — Pareto Front (4 mục tiêu)", fontweight='bold')
    ax_pc.legend(fontsize=9)
    ax_pc.grid(True, alpha=0.3)

    plt.savefig('/mnt/user-data/outputs/bai07_pareto_front.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Pareto Front đã lưu: bai07_pareto_front.png")

    # ─────────────────────────────────────────
    # CÂU 7.4.3: TOPSIS trên tập Pareto → nghiệm thoả hiệp
    # ─────────────────────────────────────────
    print("\n--- Câu 7.4.3: TOPSIS chọn nghiệm thoả hiệp ---")

    # Trọng số chính sách: tăng trưởng=0.40, bao trùm=0.25, môi trường=0.20, an ninh=0.15
    w_policy = np.array([0.40, 0.25, 0.20, 0.15])
    is_benefit_f = [True, False, False, False]  # f1 benefit; f2,f3,f4 cost

    # TOPSIS trên F_display
    norms = np.sqrt((F_display**2).sum(axis=0))
    R_f = F_display / (norms + 1e-12)
    V_f = R_f * w_policy
    A_star = np.where(is_benefit_f, V_f.max(axis=0), V_f.min(axis=0))
    A_neg  = np.where(is_benefit_f, V_f.min(axis=0), V_f.max(axis=0))
    S_plus  = np.sqrt(((V_f - A_star)**2).sum(axis=1))
    S_minus = np.sqrt(((V_f - A_neg )**2).sum(axis=1))
    C_star  = S_minus / (S_plus + S_minus + 1e-12)

    best_idx = C_star.argmax()
    best_F   = F_display[best_idx]
    best_X   = X_pareto[best_idx].reshape(6, 4)

    print(f"\nNghiệm thoả hiệp (TOPSIS C*={C_star[best_idx]:.4f}):")
    print(f"  f1 GDP gain  = {best_F[0]:,.1f} tỷ VND")
    print(f"  f2 Gini MAD  = {best_F[1]:,.2f}")
    print(f"  f3 Phát thải = {best_F[2]:,.2f}")
    print(f"  f4 Rủi ro    = {best_F[3]:,.2f}")

    df_compromise = pd.DataFrame(
        best_X.round(1),
        index=['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD'],
        columns=['I(hạ tầng)', 'D(CĐS)', 'AI', 'H(nhân lực)']
    )
    print(f"\nPhân bổ thoả hiệp (tỷ VND):")
    print(df_compromise.to_string())

    # ─────────────────────────────────────────
    # CÂU 7.4.4: Chi phí cơ hội
    # ─────────────────────────────────────────
    print("\n--- Câu 7.4.4: Chi phí cơ hội giữa các nghiệm ---")

    # Nghiệm GDP cao nhất
    best_growth_idx = F_display[:, 0].argmax()
    F_growth = F_display[best_growth_idx]

    # So sánh
    delta_f1 = (F_growth[0] - best_F[0]) / F_growth[0] * 100
    delta_f2 = (F_growth[1] - best_F[1]) / (abs(best_F[1]) + 1e-9) * 100
    delta_f3 = (F_growth[2] - best_F[2]) / (abs(best_F[2]) + 1e-9) * 100

    print(f"\n  Nghiệm GDP cao nhất:")
    print(f"    f1 = {F_growth[0]:,.1f}  f2 = {F_growth[1]:.2f}  f3 = {F_growth[2]:.2f}  f4 = {F_growth[3]:.2f}")
    print(f"\n  Nghiệm thoả hiệp:")
    print(f"    f1 = {best_F[0]:,.1f}  f2 = {best_F[1]:.2f}  f3 = {best_F[2]:.2f}  f4 = {best_F[3]:.2f}")
    print(f"\n  Chi phí cơ hội khi chọn thoả hiệp thay vì max GDP:")
    print(f"    ΔZ (GDP giảm)    = {abs(F_growth[0]-best_F[0]):,.1f} tỷ ({delta_f1:.1f}%)")
    print(f"    Δf2 (công bằng cải thiện) = {(F_growth[1]-best_F[1]):.2f}")
    print(f"    Δf3 (phát thải giảm)      = {(F_growth[2]-best_F[2]):.2f}")

    # Vẽ biểu đồ so sánh
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle("Bài 7: Phân bổ và Chi phí Cơ hội", fontsize=13, fontweight='bold')

    # Heatmap phân bổ thoả hiệp
    import seaborn as sns
    sns.heatmap(best_X, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=['Hạ tầng', 'CĐS DN', 'AI', 'Nhân lực'],
                yticklabels=['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD'],
                ax=axes2[0], linewidths=0.5)
    axes2[0].set_title("Phân bổ tối ưu thoả hiệp (tỷ VND)", fontweight='bold')

    # Radar so sánh 2 nghiệm
    labels_radar = ['GDP gain\n(÷1000)', 'Công bằng\n(thấp=tốt)', 'Phát thải\n(thấp=tốt)', 'Rủi ro\n(thấp=tốt)']
    values_growth = [F_growth[0]/1000, F_growth[1], F_growth[2], max(0, F_growth[3])]
    values_comp   = [best_F[0]/1000,   best_F[1],   best_F[2],   max(0, best_F[3])]
    x_pos = np.arange(4)
    axes2[1].bar(x_pos - 0.2, values_growth, 0.4, label='Max GDP', color='#C62828', alpha=0.8)
    axes2[1].bar(x_pos + 0.2, values_comp,   0.4, label='Thoả hiệp', color='#1565C0', alpha=0.8)
    axes2[1].set_xticks(x_pos)
    axes2[1].set_xticklabels(labels_radar, fontsize=9)
    axes2[1].set_title("So sánh nghiệm Max GDP vs Thoả hiệp", fontweight='bold')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/bai07_compromise_solution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Biểu đồ thoả hiệp đã lưu: bai07_compromise_solution.png")

else:
    print("\n⚠ pymoo chưa cài. Chạy: pip install pymoo --break-system-packages")
    print("Code đã sẵn sàng, chỉ cần cài thư viện rồi chạy lại.")

print("\n--- Thảo luận chính sách ---")
print("""
a) Đánh đổi tăng trưởng ↔ bao trùm: thường rõ ràng trên Pareto front
   → Nghiệm GDP cao nhất thường phân bổ thiên về SE/RRD (AI cao)
   → Chi phí cơ hội ~5-15% GDP để đạt công bằng vùng hợp lý

b) Trọng số (0.40, 0.25, 0.20, 0.15) thiên về tăng trưởng:
   → Theo cam kết COP26: nên tăng w_f3 (môi trường) lên 0.25+
   → Theo QĐ 127: nên tăng w_f4 (an ninh dữ liệu) lên 0.20+

c) NSGA-II vs LP đơn mục tiêu:
   → LP cho 1 nghiệm tối ưu dứt khoát nhưng ẩn đi đánh đổi
   → NSGA-II cho bản đồ toàn diện các phương án — hỗ trợ ra quyết
     định chính sách tốt hơn, nhưng KHÔNG thay thế quyết định chính trị
""")

print("=" * 60)
print("HOÀN THÀNH BÀI 7")
print("=" * 60)
